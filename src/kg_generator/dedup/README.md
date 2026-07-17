# Quality and Deduplication

## Purpose

Evaluate document quality and remove repeated documents before entity extraction.

## Input → output

Documents → accepted documents, quality profiles, and duplicate-cluster decisions.

## Key files

- `quality.py` scores text, rejects empty/too-short documents, and records suspicious signals such as repeated lines as review flags.
- `near_dedup.py` detects exact and near duplicates with MinHash, SimHash, or n-grams.

## Place in pipeline

After ingest/cleaning and before curation output or KG extraction. This works on whole documents, not entities.

## Language handling

Line breaks are preserved for repeated-line checks. Vietnamese input does not use the English short-token/gibberish rule because short Vietnamese syllables are normal.
