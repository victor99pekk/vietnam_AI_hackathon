# Knowledge Graph Generator

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

End-to-end pipeline for supervised fine tuning of LLMs. From legal webscraping, Knowledge graph generation, to Question-Answer pair generation used in the SFT of a LLM.

## About the Project

Polygraph began as a hackathon project at the **[Vietnam AI Innovation Challenge](https://www.vietnamaichallenge.com)** — Vietnam's first 100% AI-native hackathon, co-organized by the National Innovation Center (NIC), Meta, and the AI for Vietnam Foundation. Built over 48 hours, it took on the real-world problem of generating high-quality, fact-grounded training data for Vietnamese LLMs.

The project won the **$5,000 USD Meta Prize** and was subsequently developed further as a research initiative under the **[AI for Vietnam Foundation](https://aiforvietnam.org)** — a non-profit dedicated to accelerating Vietnam's AI ecosystem through open datasets, training, and applied research.

<p align="center">
  <img src="figures/meta_award.png" alt="Meta Award" width="400"/>
</p>

## Installation

```bash
make install
```

This sets up a virtual environment with `uv`, installs the package with all core extras (curation, Neo4j, embeddings, dev tools), and downloads the required spaCy models.

For Docker:

```bash
docker build -t kg-gen .
```

## CLI

All commands are driven through `make`. Run `make help` for the full list.

```bash
make ingest                         # Run the full KG generation pipeline
make new-graph dataset=wikipedia    # Build KG and upload to Neo4j
make neo4j-new-graph                # Build KG directly in Neo4j (scales beyond RAM)
make eval-method1                   # Quick KG health check
make eval-all                       # Full evaluation end-to-end
make download-wikipedia wiki_lang=vi wiki_count=500
make test                           # Run the test suite
```

See [docs/usage.md](docs/usage.md) for the full command reference.

## Evaluation

The evaluation suite provides two complementary assessments:

| Method | What it measures | Runtime |
|---|---|---|
| **Method 1 — Data Quality** | Graph health (orphans, density, schema, duplicates), SFT pair quality (faithfulness, relevancy), fact coverage | Seconds (CPU) |
| **Method 2 — Model Ablation** | Does KG-structured training data produce a better model? Fine-tunes base → KG-managed → raw-text and benchmarks all three | Hours (GPU recommended) |

```bash
make eval-method1                   # Quick data quality check
make eval-all                       # Full ablation study
```

See [docs/evaluation.md](docs/evaluation.md) for details.

## Project Structure

```
src/kg_generator/           Main package
├── ingest/                 Data loading, cleaning, chunking
├── dedup/                  Document & chunk deduplication, quality filtering
├── curate/                 Dataset curation with provenance tracking
├── extract/                Entity & relation extraction (spaCy, underthesea, GraphGen/LLM)
├── resolve/                Entity resolution & deduplication
├── graph/                  Graph construction (NetworkX, Neo4j)
├── evaluate/               Evaluation suite
│   ├── data_eval/          Structural audit, SFT quality, fact coverage
│   ├── model_eval/         QA dataset generation, LoRA fine-tuning, ablation benchmarking
│   ├── graphgen/           Paper-inspired subgraph + multi-hop QA
│   └── plots/              Visualization utilities
├── export/                 JSON, GraphML, Neo4j CSV, RDF, Cytoscape.js
├── cli.py                  Command-line interface (kg-gen)
├── pipeline.py             Pipeline orchestrator
└── api.py                  FastAPI demo backend

configs/                    YAML pipeline presets
docs/                       Documentation
data/                       Sample inputs & curated outputs
tests/                      pytest test suite (71 tests)
demo/                       Interactive web demo
presentation/               Project presentation deck
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.

