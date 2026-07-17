# Extraction

## Purpose

Turn document text into entities and relationships that can become graph nodes and edges.

## Input → output

Clean documents → entity records and `(subject, predicate, object)` triples.

## Key files

- `entities.py` extracts named entities using English or Vietnamese-capable backends.
- `relations.py` extracts relationships using rules or an optional LLM.

## Place in pipeline

After document curation or quality/deduplication; before entity resolution and graph construction.
