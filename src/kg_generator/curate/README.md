# Dataset Curation

## Purpose

Create a traceable, reusable text dataset from raw documents.

## Input → output

Raw documents plus a source manifest → `curated.jsonl`, `audit.csv`, `quality_report.json`, and `dataset_manifest.json`.

## Key files

- `manifest.py` validates source, license, language, and version metadata; it also creates hashes.
- `pipeline.py` coordinates ingest, shared quality/deduplication, and artifact writing.

## Place in pipeline

The recommended preparation stage before building a KG. It uses shared logic from `dedup/`.
