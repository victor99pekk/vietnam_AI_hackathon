# Knowledge Graph Generator

A tool for generating knowledge bases to support LLM training and Graph RAG workflows, built for the Vietnam AI Challenge.

## Overview

The Knowledge Graph Generator converts raw text into structured knowledge graphs through a 5-stage pipeline:

```
Raw Data → [1. Ingest & Clean] → [2. Extract] → [3. Resolve] → [4. Build Graph] → [5. Evaluate & Export]
```

Supports English out of the box, with a Vietnamese backend available via `underthesea`.

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

> New to uv? Read the [UV Setup Guide](docs/UV_SETUP.md) — it's written for beginners.

## Pipeline Stages

| Stage | Description |
|---|---|
| **Ingest** | Load from TXT, JSON, CSV, JSONL; normalize whitespace & encoding |
| **Dedup & Quality** | MinHash/SimHash/n-gram dedup + heuristic quality filtering |
| **Extract** | spaCy NER + rule-based relation extraction (or LLM-powered) |
| **Relate** | Entity resolution via embedding similarity (multilingual) |
| **Build Graph** | Directed graph construction + ontology validation |
| **Evaluate & Export** | 7 quality metrics + JSON, GraphML, Neo4j CSV, RDF, Cytoscape.js |

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

