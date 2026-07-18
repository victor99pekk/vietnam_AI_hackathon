# Knowledge Graph Generator

A tool for generating knowledge bases to support LLM training and Graph RAG workflows, built for the Vietnam AI Challenge.

## Overview

The Knowledge Graph Generator converts raw text into structured knowledge graphs through a 5-stage pipeline:

```
Raw Data → [1. Ingest & Clean] → [2. Extract] → [3. Resolve] → [4. Build Graph] → [5. Evaluate & Export]
```

Supports English out of the box. Vietnamese supports both DeepSeek GraphGen
extraction and an offline `underthesea` backend while keeping one graph schema.

## Quick Start

```bash
# Using uv (recommended — see docs/UV_SETUP.md for details)
uv venv
source .venv/bin/activate
uv pip install -e "."
python -m spacy download en_core_web_sm

# Run
kg-gen quick -i data/sample/
```

Vietnamese GraphGen and offline runs:

```bash
uv sync --extra vi --extra llm
kg-gen run -c configs/vietnamese.yaml
kg-gen run -c configs/vietnamese.yaml --no-llm
```

Selectable Vietnamese demo presets:

```bash
# Fast: sentence chunks + MinHash/exact dedup + Underthesea
kg-gen run -c configs/vietnamese_fast.yaml -o output/vi-fast

# Balanced: sentence chunks + two-level MinHash + GraphGen
kg-gen run -c configs/vietnamese.yaml -o output/vi-balanced

# Quality: semantic chunks + layered/semantic dedup + GraphGen + embeddings
uv sync --extra embeddings --extra vi --extra llm
kg-gen run -c configs/vietnamese_quality.yaml -o output/vi-quality
```

Scraped JSONL works with the same strategy framework:

```bash
uv sync --extra scraper --extra embeddings --extra vi --extra llm
kg-gen scrape --seed-file data/download_data/seeds/vietnamese_sources.txt \
  -o data/scraped/vietnamese_demo
kg-gen run -c configs/vietnamese_scraped.yaml \
  -i data/scraped/vietnamese_demo -o output/vi-scraped
```

> New to uv? Read the [UV Setup Guide](docs/UV_SETUP.md) — it's written for beginners.

## Pipeline Stages

| Stage | Description |
|---|---|
| **Ingest** | Load from TXT, JSON, CSV, JSONL; normalize whitespace & encoding |
| **Chunk** | Select fixed/paragraph, sentence-aware, or multilingual semantic boundaries |
| **Dedup & Quality** | Document + chunk exact/MinHash/SimHash/n-gram/semantic/layered dedup |
| **Extract** | spaCy NER + rule-based relation extraction (or LLM-powered) |
| **Relate** | Entity resolution via embedding similarity (multilingual) |
| **Build Graph** | Directed graph construction + ontology validation |
| **Evaluate & Export** | 7 quality metrics + JSON, GraphML, Neo4j CSV, RDF, Cytoscape.js |

## Dataset Curation Toolkit

`kg-gen curate` builds an auditable English or Vietnamese text dataset for LLM work. It keeps the KG workflow intact and writes canonical curated JSONL, token-budget training shards, source/record audits, duplicate evidence, reports, and a provenance manifest with hashes.

```bash
uv pip install -e ".[curation]"
kg-gen curate -i data/sample/ -m configs/example_source_manifest.yaml --device cuda
```

See [the curation guide](docs/dataset_curation.md) for the source-manifest format, outputs, and demo procedure.
See [deduplication experiments](docs/deduplication_experiments.md) for MinHash and semantic experiment commands.
See the [Wikimedia Wikipedia pilot guide](docs/wikipedia_dataset.md) to stream English or Vietnamese samples into the pipeline.

## Component Guides

The [package architecture guide](src/kg_generator/README.md) explains how the curation and KG workflows fit together. Each pipeline component also has a short README describing its responsibility, inputs, outputs, and key files.

## Project Structure

```
hackathon/
├── pyproject.toml
├── src/kg_generator/
│   ├── cli.py              # CLI (click)
│   ├── config.py           # Configuration & ontology
│   ├── pipeline.py         # Pipeline orchestrator
│   ├── ingest/             # Data loading & text cleaning
│   ├── dedup/              # Near-dedup & quality filtering
│   ├── extract/            # Entity + relation extraction
│   ├── resolve/            # Entity resolution
│   ├── graph/              # Graph building & enrichment
│   ├── evaluate/           # Quality metrics
│   └── export/             # Multi-format export
├── configs/                # Ontology & pipeline YAML configs
├── tests/                  # Pytest test suite
├── docs/                   # Usage documentation
└── data/sample/            # Sample input files
```

## License

MIT
