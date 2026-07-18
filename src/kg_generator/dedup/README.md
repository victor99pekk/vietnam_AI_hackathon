# Quality and Deduplication

## Purpose

Evaluate document quality and remove repeated documents before entity extraction.

## Input → output

Documents → accepted documents, quality profiles, and duplicate-cluster decisions.

## Key files

- `quality.py` scores text, rejects empty/too-short documents, and records suspicious signals such as repeated lines as review flags.
- `near_dedup.py` detects exact and near duplicates with exact hashes, MinHash,
  SimHash, n-grams, multilingual semantic similarity, or a layered
  MinHash-then-semantic strategy.

## Place in pipeline

The KG pipeline can run deduplication twice: once on cleaned source documents,
then again on the chunks produced for extraction. Entity duplication is handled
separately by the resolver.

## Language handling

Line breaks are preserved for repeated-line checks. Vietnamese input does not use the English short-token/gibberish rule because short Vietnamese syllables are normal.
