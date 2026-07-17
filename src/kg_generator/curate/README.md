# Dataset Curation

## Purpose

Create a traceable, reusable text dataset from raw documents.

## Input → output

Raw documents plus a source manifest → canonical `curated.jsonl`, token-budget training shards, source and record audits, duplicate evidence, reports, and a provenance manifest.

## Key files

- `manifest.py` validates source, license, language, and version metadata; it also creates hashes.
- `pipeline.py` coordinates normalization, quality, layered deduplication, record splitting, batching, and immutable artifact writing.
- `processing.py` owns language-aware segmentation, BGE-M3 token counting, and batched semantic review.

## Place in pipeline

The recommended preparation stage before building a KG. It uses shared logic from `dedup/`.
