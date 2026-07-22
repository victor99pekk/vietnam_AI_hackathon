"""Tests for the local Wikimedia sample writer without network access."""

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "download_wikipedia.py"
SPEC = importlib.util.spec_from_file_location("wikipedia_downloader", SCRIPT_PATH)
assert SPEC and SPEC.loader
wikipedia_downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wikipedia_downloader)


def test_write_sample_and_manifest(tmp_path):
    articles = iter([
        {"id": "empty", "text": "", "title": "Empty", "url": "https://example.test/empty"},
        {"id": "article", "text": "A useful article.", "title": "Article", "url": "https://example.test/article"},
    ])
    output_path = tmp_path / "sample.jsonl"
    manifest_path = tmp_path / "sample_manifest.yaml"

    assert wikipedia_downloader.write_sample(articles, 1, output_path) == 1
    wikipedia_downloader.write_manifest(manifest_path, "vi", "20231101", 1)

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["id"] == "article"
    assert record["title"] == "Article"
    manifest = manifest_path.read_text(encoding="utf-8")
    assert "language: vi" in manifest
    assert "version: 20231101-vi-1" in manifest
