# Entity Resolution

## Purpose

Merge different mentions of the same real-world entity.

## Input → output

Extracted entity records → resolved, unique entity records.

## Key files

- `resolver.py` compares entity names with string similarity or optional embeddings.

## Place in pipeline

After extraction and before graph construction. This is entity-level matching, not document deduplication.
