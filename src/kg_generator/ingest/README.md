# Ingest

## Purpose

Load raw text data and normalize it before quality checks or KG extraction.

## Input → output

TXT, JSON, JSONL, or CSV files → `Document` objects with text, source, ID, and metadata.

## Key files

- `loader.py` reads supported file formats and creates `Document` records.
- `cleaner.py` normalizes whitespace and punctuation with English and Vietnamese-ready backends.

## Place in pipeline

First step for both dataset curation and direct KG generation.
