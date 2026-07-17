# Quality and Deduplication

## Purpose

Evaluate document quality and remove repeated documents before entity extraction.

## Input → output

Documents → accepted documents, quality profiles, and duplicate-cluster decisions.

## Key files

- `quality.py` scores text and explains rejection reasons such as short content or repeated lines.
- `near_dedup.py` detects exact and near duplicates with MinHash, SimHash, or n-grams.

## Place in pipeline

After ingest/cleaning and before curation output or KG extraction. This works on whole documents, not entities.
