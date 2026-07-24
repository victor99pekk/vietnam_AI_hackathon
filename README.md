# Polygraph: KG-Grounded SFT Data for LLMs

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🌐 Find raw documents → 🧠 Build knowledge graph → 💬 Generate QA pairs → 🎯 Fine-tune LLM

Turn unstructured text into structured knowledge graphs, then into high-quality QA pairs that measurably improve LLM performance.

## Contents

<!-- - [Polygraph: KG-Grounded SFT Data for LLMs](#polygraph-kg-grounded-sft-data-for-llms)
  - [Contents](#contents)
    - [About the Project](#about-the-project) -->
  - [Quick Start](#quick-start), [Scraping](#scraping), [Knowledge Graph Generation](#knowledge-graph-generation), [Fine-Tuning](#fine-tuning)
  - [Results](#results)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)

### About the Project

<img src="figures/meta_award.png" alt="Meta Award" width="350" align="right"/>

Polygraph began as a hackathon project at the **[Vietnam AI Innovation Challenge](https://www.vietnamaichallenge.com)** co-organized by the National Innovation Center (NIC), Meta, and the AI for Vietnam Foundation. Built over 48 hours, it took on the real-world problem of generating high-quality, fact-grounded training data for Vietnamese LLMs.

The project won the **$5,000 USD Meta Prize** and was subsequently developed further as a research initiative under the **[AI for Vietnam Foundation](https://aiforvietnam.org)**, a non-profit dedicated to accelerating Vietnam's AI ecosystem through open datasets, training, and applied research.

## Quick Start

All commands are driven through `make`. Run `make help` for the full list.

```bash
make install                        # One-time: set up venv and all dependencies
make test                           # Verify everything works
```

### Scraping

```bash
make scrape                         # Scrape web pages into JSONL
make download-wikipedia wiki_lang=vi wiki_count=500
make scrape-full                    # Full scrape → discover → re-scrape → clean
```

### Knowledge Graph Generation

```bash
make ingest                         # Run the full KG generation pipeline
make new-graph dataset=wikipedia    # Build KG and upload to Neo4j (classic)
make neo4j-new-graph                # Build KG directly in Neo4j (scales beyond RAM)
make eval                           # Structural audit, SFT pair quality, fact coverage — runs in seconds
```

### Fine-Tuning

```bash
make eval-datasets                  # Generate QA training pairs from the knowledge graph
make eval-finetune variant=kg       # Fine-tune base → KG-managed → raw-text and benchmark all three (CPU)
make eval-finetune variant=kg DEVICE=cuda  # Same, on GPU
make eval-full                      # Quality → datasets → finetune → benchmark end-to-end
```

See [docs/usage.md](docs/usage.md) for the full command reference.

## Results

We fine-tuned [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and ran an ablation study comparing three model variants on 10 Vietnamese Wikipedia articles with 50 held-out test samples:

| Metric | Base Model | KG-Trained (B) | Flat (C) | Improvement |
|---|---|---|---|---|
| **Factual Accuracy** | 2.7% | **18.1%** | 4.2% | 6.8× over base |
| **Multi-hop Accuracy** | 2.7% | **18.1%** | 4.2% | 6.8× over base |
| **Hallucination Rate** | 94% | **0%** | 36% | Eliminated entirely |
| **Consistency Score** | 0.56 | **0.83** | 0.76 | +48% |
| **Avg Response Length** | 82 words | **21 words** | 24 words | 74% shorter, more concise |

> *(B) KG-Trained: fine-tuned on QA pairs generated from the knowledge graph. (C) Flat: fine-tuned on QA pairs generated directly from the same documents, without KG structuring — not raw unformatted text.*

**Key takeaway:** KG-structured training data eliminated hallucinations and delivered 6.8× better factual accuracy. Flat fine-tuning alone barely moved the needle (4.2% vs 2.7%) — the knowledge graph structure is what matters.

> *Note: Results are from a small pilot study. Metrics are heuristic (token-overlap F1, word-count proxies) with wide confidence intervals. Full-scale evaluation with more documents and robust metrics is ongoing.*

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

