"""Tests for audited, multilingual dataset curation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from kg_generator.curate.manifest import SourceManifest
from kg_generator.curate.pipeline import CurationConfig, DatasetCurationPipeline
from kg_generator.curate.processing import CurationTextProcessor, SemanticReviewer, TextSpan, split_text_to_token_limit
from kg_generator.dedup.near_dedup import GlobalDeduplicator, SemanticDeduplicator
from kg_generator.dedup.quality import QualityProfiler, QualityThresholds


def word_counter(text: str) -> int:
    return len(text.split()) + 2


def write_manifest(path: Path, *, language: str = "en", version: str = "v1") -> Path:
    manifest = path / "manifest.yaml"
    manifest.write_text(
        f"dataset_name: test-corpus\nversion: {version}\nlicense: CC-BY-4.0\n"
        f"source: self-authored test data\nlanguage: {language}\n",
        encoding="utf-8",
    )
    return manifest


def curation_config(tmp_path: Path, data: Path, **overrides: object) -> CurationConfig:
    defaults: dict[str, object] = {
        "input_paths": (data,),
        "output_root": tmp_path / "output",
        "source_manifest_path": write_manifest(tmp_path),
        "token_counter": word_counter,
        "semantic_review_enabled": False,
        "max_record_tokens": 32,
        "embedding_batch_token_budget": 32,
        "shard_token_budget": 64,
    }
    defaults.update(overrides)
    return CurationConfig(**defaults)  # type: ignore[arg-type]


def test_source_manifest_requires_legal_provenance_and_defaults_language(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text("dataset_name: test\nversion: v1\nlicense: MIT\nsource: local\n", encoding="utf-8")
    assert SourceManifest.from_file(path).language == "en"

    path.write_text("dataset_name: test\nversion: v1\nsource: local\n", encoding="utf-8")
    with pytest.raises(ValueError, match="license"):
        SourceManifest.from_file(path)


def test_quality_profile_returns_explainable_reasons():
    profile = QualityProfiler(QualityThresholds(min_chars=50, min_words=10)).profile("Bad!")
    assert not profile.accepted
    assert {"too_short_characters", "too_short_words"}.issubset(profile.reasons)


def test_quality_profile_flags_repeated_lines_without_rejecting_document():
    text = "Repeated line for review.\nRepeated line for review.\nA different line keeps this document useful."
    profile = QualityProfiler(QualityThresholds(min_chars=20, min_words=3)).profile(text)

    assert profile.accepted
    assert profile.requires_review
    assert profile.repeated_line_ratio == 2 / 3
    assert "repeated_lines" in profile.review_flags


def test_vietnamese_profile_does_not_apply_english_short_token_heuristic():
    text = "và của là ở thì tôi bạn nó em anh chị này kia đó một hai ba bốn năm sáu bảy tám chín"
    profile = QualityProfiler(QualityThresholds(min_chars=20, min_words=3)).profile(text, language="vi")

    assert profile.accepted
    assert profile.short_token_ratio is None
    assert "short_token_gibberish" not in profile.review_flags


def test_vietnamese_normalization_preserves_diacritics():
    processor = CurationTextProcessor("vi")
    normalized = processor.normalize("  Dữ\u0303 liệu tiếng Việt.\r\n\r\nGiữ nguyên dấu.  ")

    assert "tiếng Việt" in normalized
    assert "Dữ" in normalized
    assert normalized == "Dữ liệu tiếng Việt.\n\nGiữ nguyên dấu."


def test_sentence_split_preserves_every_character_and_token_limit():
    class FakeProcessor:
        def sentence_spans(self, text: str):
            return [TextSpan(0, 12), TextSpan(13, 25), TextSpan(26, len(text))]

    text = "Alpha one. Beta two. Gamma three."
    pieces = split_text_to_token_limit(text, FakeProcessor(), word_counter, max_tokens=5)

    assert "".join(text[span.start:span.end] for span, _ in pieces) == text
    assert all(token_count <= 5 for _, token_count in pieces)


def test_global_dedup_selects_best_quality_then_stable_id():
    records = [
        {"doc_id": "z", "content": "same content", "quality_score": 0.8},
        {"doc_id": "a", "content": "same content", "quality_score": 0.8},
    ]
    assignments = GlobalDeduplicator().cluster(records)
    assert assignments["a"].canonical_id == "a"
    assert assignments["z"].is_duplicate


def test_semantic_dedup_logs_embedding_match_without_model_download():
    records = [
        {"doc_id": "first", "content": "Marie Curie discovered radium.", "quality_score": 0.8},
        {"doc_id": "paraphrase", "content": "Radium was discovered by Marie Curie.", "quality_score": 0.7},
        {"doc_id": "different", "content": "The weather is sunny today.", "quality_score": 0.9},
    ]
    embeddings = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    deduplicator = SemanticDeduplicator(threshold=0.95, encoder=lambda texts: embeddings)
    assignments = deduplicator.cluster(records)

    assert assignments["paraphrase"].is_duplicate
    assert assignments["paraphrase"].method == "semantic_cosine"
    assert assignments["paraphrase"].similarity > 0.99
    assert len(deduplicator.last_matches) == 1


def test_curation_generates_shards_and_reconcilable_artifacts(tmp_path):
    first = tmp_path / "first.jsonl"
    first.write_text(
        "{\"id\": \"a\", \"text\": \"A reliable data pipeline records the origin and license of every training document for future audits.\", \"category\": \"guide\"}\n"
        "{\"id\": \"vi\", \"text\": \"Dữ liệu tiếng Việt cần được xử lý bằng Unicode để giữ nguyên dấu câu và ký tự hợp lệ trong mọi báo cáo.\"}\n",
        encoding="utf-8",
    )
    second = tmp_path / "second.jsonl"
    second.write_text(
        "{\"id\": \"b\", \"text\": \"A reliable data pipeline records the origin and license of every training document for future audits.\"}\n"
        "{\"id\": \"short\", \"text\": \"Too short.\"}\n",
        encoding="utf-8",
    )
    config = curation_config(tmp_path, first, input_paths=(first, second))
    output_dir = DatasetCurationPipeline(config).execute()

    assert {path.name for path in output_dir.iterdir()} == {
        "curated.jsonl", "audit.csv", "record_audit.csv", "duplicate_matches.csv",
        "quality_report.json", "batch_manifest.json", "dataset_manifest.json", "shards",
    }
    curated = [json.loads(line) for line in (output_dir / "curated.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(curated) == 2
    assert any("Dữ liệu" in item["text"] for item in curated)
    assert all("parent_document_id" in item for item in curated)
    with (output_dir / "audit.csv").open(encoding="utf-8") as handle:
        audit = list(csv.DictReader(handle))
    report = json.loads((output_dir / "quality_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    batches = json.loads((output_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    assert len(audit) == report["source_record_counts"]["input"] == 4
    assert manifest["dataset"]["license"] == "CC-BY-4.0"
    assert {entry["path"] for entry in batches["shards"]} == {"shards/batch-00000.jsonl"}
    assert batches["shards"][0]["token_count"] <= 64

    with pytest.raises(FileExistsError):
        DatasetCurationPipeline(config).execute()


def test_semantic_candidates_are_review_only_and_oom_batches_retry(tmp_path):
    data = tmp_path / "records.jsonl"
    data.write_text(
        "{\"id\": \"first\", \"text\": \"Marie Curie discovered radium and documented the scientific work for future researchers.\"}\n"
        "{\"id\": \"second\", \"text\": \"Radium was discovered by Marie Curie and the research was documented for later scientists.\"}\n",
        encoding="utf-8",
    )
    calls: list[int] = []

    def encoder(texts: list[str]):
        calls.append(len(texts))
        if len(texts) > 1:
            raise RuntimeError("CUDA out of memory")
        return [[1.0, 0.0] if "Marie Curie" in text else [0.99, 0.01] for text in texts]

    config = curation_config(
        tmp_path,
        data,
        semantic_review_enabled=True,
        semantic_encoder=encoder,
        semantic_review_threshold=0.95,
        embedding_batch_token_budget=100,
    )
    output_dir = DatasetCurationPipeline(config).execute()
    rows = list(csv.DictReader((output_dir / "record_audit.csv").open(encoding="utf-8")))

    assert calls[0] == 2
    assert calls.count(1) == 2
    assert all(row["decision"] == "accepted" for row in rows)
    assert all("semantic_duplicate_candidate" in row["review_flags"] for row in rows)
    matches = list(csv.DictReader((output_dir / "duplicate_matches.csv").open(encoding="utf-8")))
    assert any(row["decision"] == "review_only" for row in matches)


def test_resume_reuses_embedding_cache_after_interruption(tmp_path):
    data = tmp_path / "records.jsonl"
    data.write_text(
        "{\"id\": \"one\", \"text\": \"First independently useful document contains enough words for the curation acceptance threshold.\"}\n"
        "{\"id\": \"two\", \"text\": \"Second independently useful document contains enough words for the curation acceptance threshold.\"}\n",
        encoding="utf-8",
    )
    attempts = 0

    def failing_encoder(texts: list[str]):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("simulated interruption")
        return [[1.0, 0.0] for _ in texts]

    config = curation_config(
        tmp_path, data, semantic_review_enabled=True, semantic_encoder=failing_encoder,
        embedding_batch_token_budget=16, max_record_tokens=16, shard_token_budget=32,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        DatasetCurationPipeline(config).execute()

    resumed_calls: list[list[str]] = []

    def succeeding_encoder(texts: list[str]):
        resumed_calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    resumed = CurationConfig(
        **{**config.__dict__, "resume": True, "semantic_encoder": succeeding_encoder}
    )
    output_dir = DatasetCurationPipeline(resumed).execute()
    assert output_dir.exists()
    assert len(resumed_calls) == 1


def test_curate_cli_wires_requested_configuration(monkeypatch, tmp_path):
    from kg_generator import cli

    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def execute(self):
            return tmp_path / "out"

    input_path = tmp_path / "input.txt"
    input_path.write_text("text", encoding="utf-8")
    manifest = write_manifest(tmp_path)
    monkeypatch.setattr(cli, "DatasetCurationPipeline", FakePipeline)
    result = CliRunner().invoke(
        cli.main,
        ["curate", "-i", str(input_path), "-m", str(manifest), "--no-semantic-review", "--device", "cpu"],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.semantic_review_enabled is False
    assert config.device == "cpu"


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CURATION_GPU_TEST") != "1",
    reason="Set RUN_CURATION_GPU_TEST=1 on a CUDA worker to download and run BGE-M3.",
)
def test_bge_m3_gpu_semantic_review_integration(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("faiss")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    texts = [
        "Marie Curie discovered radium and changed modern physics.",
        "Radium was discovered by Marie Curie, transforming modern physics.",
    ]
    records = [
        {
            "record_id": f"record-{index}",
            "text": text,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "token_count": word_counter(text),
        }
        for index, text in enumerate(texts)
    ]
    reviewer = SemanticReviewer(
        model_name="BAAI/bge-m3",
        model_revision=None,
        device="cuda",
        threshold=0.80,
        top_k=1,
        batch_token_budget=256,
    )
    matches = reviewer.review(records, tmp_path / "embeddings.npz")
    assert matches
