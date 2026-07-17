"""Tests for the auditable dataset curation workflow."""

import csv
import json
from pathlib import Path

import pytest

from kg_generator.dedup.near_dedup import GlobalDeduplicator
from kg_generator.dedup.quality import QualityProfiler, QualityThresholds
from kg_generator.curate.manifest import SourceManifest
from kg_generator.curate.pipeline import CurationConfig, DatasetCurationPipeline


def write_manifest(path: Path, *, language: str = "en") -> Path:
    manifest = path / "manifest.yaml"
    manifest.write_text(
        "dataset_name: test-corpus\nversion: v1\nlicense: CC-BY-4.0\n"
        f"source: self-authored test data\nlanguage: {language}\n",
        encoding="utf-8",
    )
    return manifest


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


def test_global_dedup_selects_best_quality_then_stable_id():
    records = [
        {"doc_id": "z", "content": "same content", "quality_score": 0.8},
        {"doc_id": "a", "content": "same content", "quality_score": 0.8},
    ]
    assignments = GlobalDeduplicator().cluster(records)
    assert assignments["a"].canonical_id == "a"
    assert assignments["z"].is_duplicate


def test_curation_generates_reconcilable_artifacts_and_preserves_unicode(tmp_path):
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
    pipeline = DatasetCurationPipeline(CurationConfig(
        input_paths=(first, second), output_root=tmp_path / "output", source_manifest_path=write_manifest(tmp_path),
    ))
    output_dir = pipeline.execute()

    assert {path.name for path in output_dir.iterdir()} == {"curated.jsonl", "audit.csv", "quality_report.json", "dataset_manifest.json"}
    curated = [json.loads(line) for line in (output_dir / "curated.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(curated) == 2
    assert any("Dữ liệu" in item["text"] for item in curated)
    with open(output_dir / "audit.csv", encoding="utf-8") as handle:
        audit = list(csv.DictReader(handle))
    report = json.loads((output_dir / "quality_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(audit) == report["record_counts"]["input"] == 4
    assert sum(row["decision"] == "accepted" for row in audit) == report["record_counts"]["accepted"]
    assert manifest["dataset"]["license"] == "CC-BY-4.0"
    assert set(manifest["artifacts"]) == {"curated.jsonl", "audit.csv", "quality_report.json"}

    with pytest.raises(FileExistsError):
        pipeline.execute()
