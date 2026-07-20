# Knowledge Graph Generator

A compact pipeline for converting raw text into structured knowledge graphs for LLM training, dataset curation, and Graph RAG.

## What it does

- Ingests raw text from JSONL, CSV, TXT and other sources
- Creates entities, relations, provenance, and graph exports
- Supports English and Vietnamese workflows
- Offers optional LLM-assisted extraction and embedding-based entity resolution
- Exports results as JSON, GraphML, Neo4j CSV, RDF, and Cytoscape-ready data

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
kg-gen quick -i data/sample/ -o output/demo
```

## Vietnamese demo

```bash
pip install -e .[vi]
kg-gen run -c configs/vietnamese.yaml -o output/vi-balanced
```

## Optional extras

```bash
pip install -e .[embeddings]
kg-gen run -c configs/vietnamese_quality.yaml -o output/vi-quality
```

## Core features

- Modular graph pipeline: ingest, chunk, dedup, extract, resolve, export
- Quality filtering and deduplication for noisy text sources
- Vietnamese support with GraphGen and offline `underthesea` extraction
- Configurable presets in `configs/`
- Export to common graph and dataset formats

## Repo structure

- `src/kg_generator/` — main pipeline, CLI, and core modules
- `configs/` — YAML presets for demo and Vietnamese workflows
- `docs/` — usage guides and project documentation
- `data/` — sample inputs, scraped sources, and download utilities
- `tests/` — pytest test coverage
- `pitch/` — presentation assets and HTML demos

## Useful commands

```bash
kg-gen run -c configs/vietnamese_fast.yaml -o output/vi-fast
kg-gen run -c configs/vietnamese_scraped.yaml -i data/scraped/vietnamese_demo
```

## License

MIT
