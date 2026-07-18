# Ingest

## Purpose

Load raw text data and normalize it before quality checks or KG extraction.

## Input → output

TXT, JSON, JSONL, or CSV files → `Document` objects with text, source, ID, and metadata.

## Key files

- `loader.py` reads supported file formats and creates `Document` records.
- `cleaner.py` normalizes whitespace and punctuation with English and Vietnamese-ready backends.
- `chunker.py` provides fixed-character, sentence-aligned, and multilingual semantic strategies.

Vietnamese sentence chunking uses Underthesea when installed and a Unicode
sentence-boundary fallback for GraphGen-only environments. Semantic chunking
uses sentence embeddings to open a boundary at topic shifts while retaining a
maximum token budget and complete-sentence overlap.

## Place in pipeline

First step for both dataset curation and direct KG generation.
