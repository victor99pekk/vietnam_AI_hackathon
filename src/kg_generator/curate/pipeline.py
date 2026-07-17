"""End-to-end, auditable, Vietnamese-ready dataset curation."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_generator.curate.manifest import SourceManifest, sha256_file, sha256_text, stable_json_hash
from kg_generator.curate.processing import (
    DEFAULT_BGE_MODEL,
    BgeTokenCounter,
    CurationTextProcessor,
    SemanticReviewer,
    split_text_to_token_limit,
)
from kg_generator.dedup.near_dedup import DuplicateMatch, GlobalDeduplicator
from kg_generator.dedup.quality import QualityProfiler, QualityThresholds
from kg_generator.ingest.loader import DataLoader


@dataclass(frozen=True)
class CurationConfig:
    """Configuration for an immutable curated dataset version."""

    input_paths: tuple[Path, ...]
    output_root: Path
    source_manifest_path: Path
    surface_dedup_threshold: float = 0.90
    semantic_review_threshold: float = 0.92
    semantic_model: str = DEFAULT_BGE_MODEL
    semantic_model_revision: str | None = None
    semantic_review_enabled: bool = True
    semantic_top_k: int = 20
    device: str = "cuda"
    max_record_tokens: int = 2_048
    embedding_batch_token_budget: int = 8_192
    shard_token_budget: int = 1_000_000
    min_chars: int = 50
    min_words: int = 10
    resume: bool = False
    # Compatibility inputs from the first curation prototype. They no longer
    # switch between mutually exclusive deduplication implementations.
    dedup_threshold: float | None = None
    dedup_method: str | None = None
    token_counter: Callable[[str], int] | None = field(default=None, repr=False, compare=False)
    semantic_encoder: Callable[[list[str]], Sequence[Sequence[float]]] | None = field(
        default=None, repr=False, compare=False
    )


class DatasetCurationPipeline:
    """Build an immutable curated corpus with source and record-level audits."""

    def __init__(self, config: CurationConfig) -> None:
        self.config = config
        self.source_manifest = SourceManifest.from_file(config.source_manifest_path)
        self._validate_config()
        self.output_dir = (
            config.output_root
            / self._safe_name(self.source_manifest.dataset_name)
            / self._safe_name(self.source_manifest.version)
        )
        self.stage_dir = self.output_dir.parent / f".{self.output_dir.name}.staging"
        self.processor = CurationTextProcessor(self.source_manifest.language)
        self.profiler = QualityProfiler(
            QualityThresholds(min_chars=config.min_chars, min_words=config.min_words)
        )
        self.surface_threshold = (
            config.dedup_threshold
            if config.dedup_threshold is not None
            else config.surface_dedup_threshold
        )
        self.surface_deduplicator = GlobalDeduplicator(
            threshold=self.surface_threshold,
            shingle_fn=self.processor.word_shingles,
        )
        self._bge_token_counter: BgeTokenCounter | None = None
        if config.token_counter is None:
            self._bge_token_counter = BgeTokenCounter(config.semantic_model, config.semantic_model_revision)
            self.count_tokens = self._bge_token_counter.count
        else:
            self.count_tokens = config.token_counter
        self._semantic_batch_sizes: list[int] = []
        self._run_configuration_hash = self._configuration_hash()

    def execute(self) -> Path:
        """Run curation, atomically publishing only a complete dataset version."""
        self._prepare_stage()
        work_dir = self.stage_dir / ".work"
        try:
            documents = self._load_and_profile_documents()
            self._apply_document_deduplication(documents)
            records = self._split_accepted_documents(documents)
            record_exact_matches = self._apply_record_exact_deduplication(records)
            semantic_matches = self._apply_semantic_review(records, work_dir / "embeddings.npz")
            accepted_records = [record for record in records if record["decision"] == "accepted"]
            self._update_source_record_counts(documents, accepted_records)
            artifacts = self._write_artifacts(
                documents=documents,
                records=records,
                accepted_records=accepted_records,
                document_matches=self.surface_deduplicator.last_matches,
                record_exact_matches=record_exact_matches,
                semantic_matches=semantic_matches,
            )
            self._write_manifest(documents, records, accepted_records, artifacts)
            shutil.rmtree(work_dir, ignore_errors=True)
            (self.stage_dir / "run_state.json").unlink(missing_ok=True)
            self.stage_dir.replace(self.output_dir)
            return self.output_dir
        except Exception:
            # Preserve staging and embedding checkpoints for --resume.
            raise

    def _validate_config(self) -> None:
        if not 0 <= self.config.surface_dedup_threshold <= 1:
            raise ValueError("surface_dedup_threshold must be between 0 and 1.")
        if self.config.dedup_threshold is not None and not 0 <= self.config.dedup_threshold <= 1:
            raise ValueError("dedup_threshold must be between 0 and 1.")
        if not 0 <= self.config.semantic_review_threshold <= 1:
            raise ValueError("semantic_review_threshold must be between 0 and 1.")
        if self.config.max_record_tokens < 3:
            raise ValueError("max_record_tokens must be at least 3.")
        if self.config.embedding_batch_token_budget < self.config.max_record_tokens:
            raise ValueError("embedding_batch_token_budget must be at least max_record_tokens.")
        if self.config.shard_token_budget < self.config.max_record_tokens:
            raise ValueError("shard_token_budget must be at least max_record_tokens.")
        if self.config.semantic_top_k < 1:
            raise ValueError("semantic_top_k must be at least 1.")
        if self.config.dedup_method not in {None, "minhash", "semantic", "layered"}:
            raise ValueError("dedup_method is legacy; use 'minhash', 'semantic', or 'layered'.")

    def _prepare_stage(self) -> None:
        if self.output_dir.exists():
            raise FileExistsError(f"Dataset version already exists: {self.output_dir}")
        state_path = self.stage_dir / "run_state.json"
        if self.stage_dir.exists():
            if not self.config.resume:
                raise FileExistsError(
                    f"Incomplete staging version exists: {self.stage_dir}. Re-run with --resume or remove it."
                )
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Cannot safely resume staging version: {self.stage_dir}") from error
            if state.get("configuration_hash") != self._run_configuration_hash:
                raise RuntimeError("Cannot resume with different curation settings or source manifest.")
            return
        self.stage_dir.mkdir(parents=True)
        state_path.write_text(
            json.dumps({"configuration_hash": self._run_configuration_hash}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _load_and_profile_documents(self) -> list[dict[str, Any]]:
        raw_documents = sorted(
            DataLoader().load(list(self.config.input_paths)),
            key=lambda document: (str(document.source), str(document.doc_id), str(document.content)),
        )
        records: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for document in raw_documents:
            content = self.processor.normalize(document.content if isinstance(document.content, str) else "")
            base_id = document.doc_id or f"doc-{sha256_text(f'{document.source}\n{content}')[:16]}"
            stable_id = str(base_id)
            collision = 1
            while stable_id in used_ids:
                stable_id = f"{base_id}-{sha256_text(f'{document.source}\n{content}\n{collision}')[:8]}"
                collision += 1
            used_ids.add(stable_id)
            words = self.processor.word_tokens(content) if content else []
            profile = self.profiler.profile(content, language=self.source_manifest.language, tokens=words)
            records.append({
                "doc_id": stable_id,
                "content": content,
                "source": str(document.source),
                "metadata": dict(document.metadata),
                "content_hash": sha256_text(content),
                "quality_score": profile.score,
                "quality_rejection_reasons": list(profile.rejection_reasons),
                "quality_review_flags": list(profile.review_flags),
                "char_count": profile.char_count,
                "word_count": profile.word_count,
                "symbol_ratio": profile.symbol_ratio,
                "repeated_line_ratio": profile.repeated_line_ratio,
                "short_token_ratio": profile.short_token_ratio,
                "decision": "rejected" if not profile.accepted else "accepted",
                "reasons": list(profile.rejection_reasons),
                "duplicate_cluster_id": "",
                "canonical_id": "",
                "dedup_method": "",
                "dedup_similarity": "",
                "matched_record_id": "",
                "output_record_count": 0,
            })
        return records

    def _apply_document_deduplication(self, documents: list[dict[str, Any]]) -> None:
        eligible = [document for document in documents if document["decision"] == "accepted"]
        assignments = self.surface_deduplicator.cluster(eligible)
        for document in eligible:
            assignment = assignments[str(document["doc_id"])]
            document["duplicate_cluster_id"] = assignment.cluster_id or ""
            document["canonical_id"] = assignment.canonical_id or ""
            document["dedup_method"] = assignment.method or ""
            document["dedup_similarity"] = assignment.similarity if assignment.similarity is not None else ""
            document["matched_record_id"] = assignment.matched_record_id or ""
            if assignment.is_duplicate:
                document["decision"] = "rejected"
                document["reasons"].append("near_duplicate")

    def _split_accepted_documents(self, documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for document in documents:
            if document["decision"] != "accepted":
                continue
            spans = split_text_to_token_limit(
                str(document["content"]),
                self.processor,
                self.count_tokens,
                self.config.max_record_tokens,
            )
            for index, (span, token_count) in enumerate(spans):
                record_id = str(document["doc_id"]) if len(spans) == 1 else f"{document['doc_id']}:part-{index:04d}"
                text = str(document["content"])[span.start:span.end]
                records.append({
                    "record_id": record_id,
                    "parent_document_id": document["doc_id"],
                    "segment_index": index,
                    "segment_count": len(spans),
                    "char_start": span.start,
                    "char_end": span.end,
                    "text": text,
                    "source": document["source"],
                    "metadata": dict(document["metadata"]),
                    "content_hash": sha256_text(text),
                    "quality_score": document["quality_score"],
                    "token_count": token_count,
                    "review_flags": list(document["quality_review_flags"]),
                    "decision": "accepted",
                    "reasons": [],
                    "canonical_id": "",
                    "dedup_method": "",
                    "dedup_similarity": "",
                    "matched_record_id": "",
                })
        return records

    @staticmethod
    def _apply_record_exact_deduplication(records: Sequence[dict[str, Any]]) -> list[DuplicateMatch]:
        canonical_by_hash: dict[str, dict[str, Any]] = {}
        matches: list[DuplicateMatch] = []
        for record in sorted(records, key=lambda item: (-float(item["quality_score"]), str(item["record_id"]))):
            content_hash = str(record["content_hash"])
            canonical = canonical_by_hash.get(content_hash)
            if canonical is None:
                canonical_by_hash[content_hash] = record
                continue
            record["decision"] = "rejected"
            record["reasons"].append("exact_duplicate_record")
            record["canonical_id"] = canonical["record_id"]
            record["dedup_method"] = "exact_hash"
            record["dedup_similarity"] = 1.0
            record["matched_record_id"] = canonical["record_id"]
            matches.append(DuplicateMatch(
                record_id=str(record["record_id"]),
                matched_record_id=str(canonical["record_id"]),
                method="exact_hash",
                similarity=1.0,
            ))
        return matches

    def _apply_semantic_review(self, records: Sequence[dict[str, Any]], cache_path: Path) -> list[DuplicateMatch]:
        accepted = [record for record in records if record["decision"] == "accepted"]
        if not self.config.semantic_review_enabled or not accepted:
            return []
        reviewer = SemanticReviewer(
            model_name=self.config.semantic_model,
            model_revision=self.config.semantic_model_revision,
            device=self.config.device,
            threshold=self.config.semantic_review_threshold,
            top_k=self.config.semantic_top_k,
            batch_token_budget=self.config.embedding_batch_token_budget,
            encoder=self.config.semantic_encoder,
        )
        matches = reviewer.review(accepted, cache_path)
        self._semantic_batch_sizes = list(reviewer.effective_batch_sizes)
        by_record: dict[str, list[DuplicateMatch]] = {}
        for match in matches:
            by_record.setdefault(match.record_id, []).append(match)
            by_record.setdefault(match.matched_record_id, []).append(match)
        for record in accepted:
            matched = by_record.get(str(record["record_id"]), [])
            if not matched:
                continue
            record["review_flags"].append("semantic_duplicate_candidate")
            strongest = sorted(matched, key=lambda match: (-match.similarity, match.matched_record_id))[0]
            record["dedup_method"] = "semantic_cosine_review"
            record["dedup_similarity"] = strongest.similarity
            record["matched_record_id"] = (
                strongest.matched_record_id
                if strongest.record_id == record["record_id"]
                else strongest.record_id
            )
        return matches

    @staticmethod
    def _update_source_record_counts(
        documents: Sequence[dict[str, Any]], accepted_records: Sequence[dict[str, Any]]
    ) -> None:
        counts = Counter(str(record["parent_document_id"]) for record in accepted_records)
        for document in documents:
            document["output_record_count"] = counts[str(document["doc_id"])]

    def _write_artifacts(
        self,
        *,
        documents: Sequence[dict[str, Any]],
        records: Sequence[dict[str, Any]],
        accepted_records: Sequence[dict[str, Any]],
        document_matches: Sequence[DuplicateMatch],
        record_exact_matches: Sequence[DuplicateMatch],
        semantic_matches: Sequence[DuplicateMatch],
    ) -> dict[str, Path]:
        curated_path = self.stage_dir / "curated.jsonl"
        canonical = sorted(accepted_records, key=lambda record: (str(record["parent_document_id"]), int(record["segment_index"])))
        self._write_jsonl(curated_path, (self._curated_payload(record) for record in canonical))

        audit_path = self.stage_dir / "audit.csv"
        source_fields = [
            "doc_id", "source", "content_hash", "quality_score", "char_count", "word_count", "symbol_ratio",
            "repeated_line_ratio", "short_token_ratio", "quality_rejection_reasons", "quality_review_flags",
            "decision", "reasons", "duplicate_cluster_id", "canonical_id", "dedup_method", "dedup_similarity",
            "matched_record_id", "output_record_count", "metadata",
        ]
        with audit_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=source_fields)
            writer.writeheader()
            for document in documents:
                writer.writerow({
                    **{field: document.get(field, "") for field in source_fields},
                    "quality_rejection_reasons": ";".join(document["quality_rejection_reasons"]),
                    "quality_review_flags": ";".join(document["quality_review_flags"]),
                    "reasons": ";".join(document["reasons"]),
                    "metadata": json.dumps(document["metadata"], ensure_ascii=False, sort_keys=True),
                })

        record_audit_path = self.stage_dir / "record_audit.csv"
        record_fields = [
            "record_id", "parent_document_id", "source", "segment_index", "segment_count", "char_start", "char_end",
            "content_hash", "quality_score", "token_count", "review_flags", "decision", "reasons", "canonical_id",
            "dedup_method", "dedup_similarity", "matched_record_id", "metadata",
        ]
        with record_audit_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=record_fields)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    **{field: record.get(field, "") for field in record_fields},
                    "review_flags": ";".join(record["review_flags"]),
                    "reasons": ";".join(record["reasons"]),
                    "metadata": json.dumps(record["metadata"], ensure_ascii=False, sort_keys=True),
                })

        matches_path = self.stage_dir / "duplicate_matches.csv"
        match_fields = ["record_id", "matched_record_id", "method", "similarity", "scope", "decision"]
        with matches_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=match_fields)
            writer.writeheader()
            for match, scope, decision in [
                *((match, "document", "auto_reject") for match in document_matches),
                *((match, "record", "auto_reject") for match in record_exact_matches),
                *((match, "record", "review_only") for match in semantic_matches),
            ]:
                writer.writerow({
                    "record_id": match.record_id,
                    "matched_record_id": match.matched_record_id,
                    "method": match.method,
                    "similarity": match.similarity,
                    "scope": scope,
                    "decision": decision,
                })

        batch_manifest_path = self._write_shards(accepted_records)
        report_path = self.stage_dir / "quality_report.json"
        report_path.write_text(
            json.dumps(self._report(documents, records, accepted_records), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "curated.jsonl": curated_path,
            "audit.csv": audit_path,
            "record_audit.csv": record_audit_path,
            "duplicate_matches.csv": matches_path,
            "batch_manifest.json": batch_manifest_path,
            "quality_report.json": report_path,
        }

    def _write_shards(self, accepted_records: Sequence[dict[str, Any]]) -> Path:
        shard_dir = self.stage_dir / "shards"
        shard_dir.mkdir()
        shuffled = sorted(
            accepted_records,
            key=lambda record: sha256_text(f"{self._run_configuration_hash}:{record['record_id']}"),
        )
        shards: list[dict[str, Any]] = []
        batch: list[dict[str, Any]] = []
        batch_tokens = 0
        for record in shuffled:
            tokens = int(record["token_count"])
            if batch and batch_tokens + tokens > self.config.shard_token_budget:
                shards.append(self._write_shard(shard_dir, len(shards), batch, batch_tokens))
                batch, batch_tokens = [], 0
            batch.append(record)
            batch_tokens += tokens
        if batch:
            shards.append(self._write_shard(shard_dir, len(shards), batch, batch_tokens))
        batch_manifest_path = self.stage_dir / "batch_manifest.json"
        batch_manifest_path.write_text(
            json.dumps({"shard_token_budget": self.config.shard_token_budget, "shards": shards}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return batch_manifest_path

    def _write_shard(
        self, shard_dir: Path, index: int, records: Sequence[dict[str, Any]], token_count: int
    ) -> dict[str, Any]:
        path = shard_dir / f"batch-{index:05d}.jsonl"
        self._write_jsonl(path, (self._curated_payload(record) for record in records))
        return {
            "path": str(path.relative_to(self.stage_dir)),
            "record_count": len(records),
            "token_count": token_count,
            "sha256": sha256_file(path),
        }

    @staticmethod
    def _write_jsonl(path: Path, payloads: Sequence[dict[str, Any]] | Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _curated_payload(record: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            **record["metadata"],
            "parent_document_id": record["parent_document_id"],
            "segment_index": record["segment_index"],
            "segment_count": record["segment_count"],
            "char_start": record["char_start"],
            "char_end": record["char_end"],
        }
        return {
            "id": record["record_id"],
            "text": record["text"],
            "source": record["source"],
            "metadata": metadata,
            "content_hash": record["content_hash"],
            "quality_score": record["quality_score"],
            "token_count": record["token_count"],
            "parent_document_id": record["parent_document_id"],
            "review_flags": record["review_flags"],
        }

    def _report(
        self,
        documents: Sequence[dict[str, Any]],
        records: Sequence[dict[str, Any]],
        accepted_records: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        source_counts = Counter(str(record["source"]) for record in accepted_records)
        token_counts = [int(record["token_count"]) for record in accepted_records]
        score_values = [float(document["quality_score"]) for document in documents]
        return {
            "source_record_counts": {
                "input": len(documents),
                "accepted": sum(document["decision"] == "accepted" for document in documents),
                "rejected": sum(document["decision"] != "accepted" for document in documents),
            },
            "record_counts": {
                "generated": len(records),
                "accepted": len(accepted_records),
                "rejected": len(records) - len(accepted_records),
            },
            "preliminary_quality": {
                "completeness": sum(bool(str(document["content"]).strip()) for document in documents) / max(len(documents), 1),
                "duplicate_rate": sum("near_duplicate" in document["reasons"] for document in documents) / max(len(documents), 1),
                "mean_quality_score": sum(score_values) / max(len(score_values), 1),
            },
            "source_composition": dict(sorted(source_counts.items())),
            "accepted_record_tokens": {
                "min": min(token_counts, default=0),
                "max": max(token_counts, default=0),
                "mean": sum(token_counts) / max(len(token_counts), 1),
            },
            "quality_review_flags": dict(sorted(Counter(
                flag for record in records for flag in record["review_flags"]
            ).items())),
            "rejection_reasons": dict(sorted(Counter(
                reason for document in documents for reason in document["reasons"]
            ).items())),
        }

    def _write_manifest(
        self,
        documents: Sequence[dict[str, Any]],
        records: Sequence[dict[str, Any]],
        accepted_records: Sequence[dict[str, Any]],
        artifacts: dict[str, Path],
    ) -> None:
        input_files = sorted({file for path in self.config.input_paths for file in self._files_for_path(path)})
        settings = self._settings()
        payload = {
            "dataset": self.source_manifest.to_dict(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_files": [{"path": str(path), "sha256": sha256_file(path)} for path in input_files],
            "record_counts": {
                "source_input": len(documents),
                "records_generated": len(records),
                "records_accepted": len(accepted_records),
            },
            "settings": settings,
            "artifacts": {
                name: {"path": path.name, "sha256": sha256_file(path)} for name, path in artifacts.items()
            },
            "configuration_hash": self._run_configuration_hash,
        }
        (self.stage_dir / "dataset_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _settings(self) -> dict[str, Any]:
        return {
            "surface_dedup_threshold": self.surface_threshold,
            "semantic_review_threshold": self.config.semantic_review_threshold,
            "semantic_review_enabled": self.config.semantic_review_enabled,
            "semantic_model": self.config.semantic_model,
            "semantic_model_revision": (
                self._bge_token_counter.resolved_revision
                if self._bge_token_counter is not None
                else self.config.semantic_model_revision or "injected"
            ),
            "semantic_top_k": self.config.semantic_top_k,
            "device": self.config.device,
            "max_record_tokens": self.config.max_record_tokens,
            "embedding_batch_token_budget": self.config.embedding_batch_token_budget,
            "embedding_effective_batches": {
                "count": len(self._semantic_batch_sizes),
                "min_records": min(self._semantic_batch_sizes, default=0),
                "max_records": max(self._semantic_batch_sizes, default=0),
            },
            "shard_token_budget": self.config.shard_token_budget,
            "quality_thresholds": asdict(self.profiler.thresholds),
            "normalization": "NFC, safe text repair, control removal, and paragraph-preserving whitespace normalization",
            "sentence_segmentation": "spaCy sentencizer" if self.source_manifest.language == "en" else "underthesea",
        }

    def _configuration_hash(self) -> str:
        input_files = sorted({file for path in self.config.input_paths for file in self._files_for_path(path)})
        return stable_json_hash({
            "dataset": self.source_manifest.to_dict(),
            "settings": {
                "surface_dedup_threshold": (
                    self.config.dedup_threshold
                    if self.config.dedup_threshold is not None
                    else self.config.surface_dedup_threshold
                ),
                "semantic_review_threshold": self.config.semantic_review_threshold,
                "semantic_review_enabled": self.config.semantic_review_enabled,
                "semantic_model": self.config.semantic_model,
                "semantic_model_revision": self.config.semantic_model_revision,
                "semantic_top_k": self.config.semantic_top_k,
                "device": self.config.device,
                "max_record_tokens": self.config.max_record_tokens,
                "embedding_batch_token_budget": self.config.embedding_batch_token_budget,
                "shard_token_budget": self.config.shard_token_budget,
                "min_chars": self.config.min_chars,
                "min_words": self.config.min_words,
            },
            "input_files": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in input_files
            ],
        })

    @staticmethod
    def _files_for_path(path: Path) -> list[Path]:
        return sorted(file for file in (path.rglob("*") if path.is_dir() else [path]) if file.is_file())

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(character if character.isalnum() or character in "-_." else "-" for character in value).strip(".-") or "dataset"
