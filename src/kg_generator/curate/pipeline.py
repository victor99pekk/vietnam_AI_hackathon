"""End-to-end, auditable dataset curation pipeline."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_generator.config import Language
from kg_generator.curate.manifest import SourceManifest, sha256_file, sha256_text, stable_json_hash
from kg_generator.dedup.near_dedup import GlobalDeduplicator
from kg_generator.dedup.quality import QualityProfiler, QualityThresholds
from kg_generator.ingest.cleaner import TextCleaner
from kg_generator.ingest.loader import DataLoader, Document


@dataclass(frozen=True)
class CurationConfig:
    input_paths: tuple[Path, ...]
    output_root: Path
    source_manifest_path: Path
    dedup_threshold: float = 0.85
    min_chars: int = 50
    min_words: int = 10


class DatasetCurationPipeline:
    """Build a curated dataset plus audit and provenance artifacts."""

    def __init__(self, config: CurationConfig) -> None:
        self.config = config
        self.source_manifest = SourceManifest.from_file(config.source_manifest_path)
        self.output_dir = config.output_root / self._safe_name(self.source_manifest.dataset_name) / self._safe_name(self.source_manifest.version)
        self.profiler = QualityProfiler(QualityThresholds(min_chars=config.min_chars, min_words=config.min_words))
        self.deduplicator = GlobalDeduplicator(threshold=config.dedup_threshold)

    def execute(self) -> Path:
        """Run curation once. Existing version directories are never overwritten."""
        if self.output_dir.exists():
            raise FileExistsError(f"Dataset version already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True)

        documents = self._load_clean_documents()
        records = self._profile_documents(documents)
        eligible = [record for record in records if not record["quality_reasons"]]
        assignments = self.deduplicator.cluster(eligible)

        for record in records:
            assignment = assignments.get(str(record["doc_id"]))
            if assignment:
                record["duplicate_cluster_id"] = assignment.cluster_id or ""
                record["canonical_id"] = assignment.canonical_id or ""
                if assignment.is_duplicate:
                    record["decision"] = "rejected"
                    record["reasons"] = ["near_duplicate"]
                else:
                    record["decision"] = "accepted"
                    record["reasons"] = []
            else:
                record["duplicate_cluster_id"] = ""
                record["canonical_id"] = ""
                record["decision"] = "rejected"
                record["reasons"] = list(record["quality_reasons"])

        accepted = [record for record in records if record["decision"] == "accepted"]
        artifacts = self._write_artifacts(records, accepted)
        self._write_manifest(records, accepted, artifacts)
        return self.output_dir

    def _load_clean_documents(self) -> list[Document]:
        loader = DataLoader()
        raw_documents = loader.load(list(self.config.input_paths))
        language = Language(self.source_manifest.language)
        cleaner = TextCleaner(language=language)
        documents: list[Document] = []
        used_ids: set[str] = set()
        for document in raw_documents:
            content = document.content if isinstance(document.content, str) else ""
            stable_id = document.doc_id or f"doc-{sha256_text(f'{document.source}\n{content}')[:16]}"
            if stable_id in used_ids:
                stable_id = f"{stable_id}-{sha256_text(f'{document.source}\n{content}')[:8]}"
            used_ids.add(stable_id)
            clean_doc = Document(
                content=content,
                source=document.source,
                doc_id=stable_id,
                metadata=dict(document.metadata),
            )
            documents.append(cleaner.clean(clean_doc))
        return documents

    def _profile_documents(self, documents: list[Document]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for document in documents:
            profile = self.profiler.profile(document.content)
            records.append({
                "doc_id": document.doc_id,
                "content": document.content,
                "source": document.source,
                "metadata": document.metadata,
                "content_hash": sha256_text(document.content),
                "quality_score": profile.score,
                "quality_reasons": list(profile.reasons),
                "char_count": profile.char_count,
                "word_count": profile.word_count,
                "symbol_ratio": profile.symbol_ratio,
                "repeated_line_ratio": profile.repeated_line_ratio,
                "short_token_ratio": profile.short_token_ratio,
            })
        return records

    def _write_artifacts(self, records: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> dict[str, Path]:
        curated_path = self.output_dir / "curated.jsonl"
        with open(curated_path, "w", encoding="utf-8") as handle:
            for record in accepted:
                payload = {
                    "id": record["doc_id"], "text": record["content"], "source": record["source"],
                    "metadata": record["metadata"], "content_hash": record["content_hash"],
                    "quality_score": record["quality_score"], "duplicate_cluster_id": record["duplicate_cluster_id"],
                }
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

        audit_path = self.output_dir / "audit.csv"
        fields = ["doc_id", "source", "content_hash", "quality_score", "char_count", "word_count", "symbol_ratio", "repeated_line_ratio", "short_token_ratio", "decision", "reasons", "duplicate_cluster_id", "canonical_id", "metadata"]
        with open(audit_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    **{field: record.get(field, "") for field in fields},
                    "reasons": ";".join(record["reasons"]),
                    "metadata": json.dumps(record["metadata"], ensure_ascii=False, sort_keys=True),
                })

        report_path = self.output_dir / "quality_report.json"
        report = self._report(records, accepted)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"curated.jsonl": curated_path, "audit.csv": audit_path, "quality_report.json": report_path}

    def _report(self, records: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(records)
        duplicate_records = sum(1 for record in records if "near_duplicate" in record["reasons"])
        source_counts = Counter(str(record["source"]) for record in accepted)
        lengths = [int(record["char_count"]) for record in accepted]
        all_grams = set()
        gram_total = 0
        clusters = Counter(str(record["duplicate_cluster_id"]) for record in records if record["duplicate_cluster_id"])
        for record in accepted:
            text = str(record["content"])
            grams = {text[index:index + 3] for index in range(max(len(text) - 2, 1))}
            all_grams.update(grams)
            gram_total += len(grams)
        score_values = [float(record["quality_score"]) for record in records]
        malformed = sum(1 for record in records if "empty_content" in record["quality_reasons"])
        return {
            "record_counts": {"input": total, "accepted": len(accepted), "rejected": total - len(accepted)},
            "preliminary_quality": {
                "completeness": sum(1 for record in records if str(record["content"]).strip()) / max(total, 1),
                "format_error_rate": malformed / max(total, 1),
                "duplicate_rate": duplicate_records / max(total, 1),
                "missing_content_rate": malformed / max(total, 1),
                "mean_quality_score": sum(score_values) / max(len(score_values), 1),
            },
            "source_composition": dict(sorted(source_counts.items())),
            "diversity": {
                "accepted_document_length": {"min": min(lengths, default=0), "max": max(lengths, default=0), "mean": sum(lengths) / max(len(lengths), 1)},
                "unique_character_ngram_ratio": len(all_grams) / max(gram_total, 1),
                "duplicate_cluster_sizes": dict(sorted((cluster, count) for cluster, count in clusters.items())),
            },
            "rejection_reasons": dict(sorted(Counter(reason for record in records for reason in record["reasons"]).items())),
        }

    def _write_manifest(self, records: list[dict[str, Any]], accepted: list[dict[str, Any]], artifacts: dict[str, Path]) -> None:
        input_files = sorted({file for path in self.config.input_paths for file in self._files_for_path(path)})
        settings = {
            "dedup_threshold": self.config.dedup_threshold,
            "quality_thresholds": asdict(self.profiler.thresholds),
            "normalization": "unicode-safe whitespace and punctuation normalization",
        }
        payload = {
            "dataset": self.source_manifest.to_dict(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_files": [{"path": str(path), "sha256": sha256_file(path)} for path in input_files],
            "record_counts": {"input": len(records), "accepted": len(accepted), "rejected": len(records) - len(accepted)},
            "settings": settings,
            "artifacts": {name: {"path": path.name, "sha256": sha256_file(path)} for name, path in artifacts.items()},
        }
        payload["configuration_hash"] = stable_json_hash({"dataset": payload["dataset"], "settings": settings, "input_files": payload["input_files"]})
        (self.output_dir / "dataset_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _files_for_path(path: Path) -> list[Path]:
        return sorted(file for file in (path.rglob("*") if path.is_dir() else [path]) if file.is_file())

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(char if char.isalnum() or char in "-_." else "-" for char in value).strip(".-") or "dataset"
