# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] — 2025-07-22

### Added
- Modular KG generation pipeline: ingest → dedup → extract → resolve → graph → evaluate → export
- Multi-format data loading (JSONL, CSV, TXT, JSON)
- MinHash, SimHash, n-gram, and semantic document deduplication
- Heuristic quality filtering for noisy text sources
- spaCy-based English NER and `underthesea`-based Vietnamese NER
- GraphGen-style LLM extraction via DeepSeek (paper-faithful Figure 8 + Figure 9)
- NetworkX in-memory graph backend and Neo4j on-disk backend
- Entity resolution (string-similarity and embedding-based)
- Export to JSON, GraphML, Neo4j CSV, RDF/Turtle, Cytoscape.js
- Evaluation suite: structural audit, SFT data quality scoring, fact coverage measurement
- Model fine-tuning ablation study: base vs. KG-managed vs. raw-text QA comparison
- CLI (`kg-gen`) with subcommands for pipeline, curation, and evaluation
- FastAPI demo backend with web frontend
- Docker support for cloud deployment
- Vietnamese language pipeline (tokenization, NER, chunking, GraphGen prompts)
- Dataset curation toolkit with provenance tracking and audit reports
- Web scraping pipeline with LLM-assisted content cleaning
- Wikipedia article downloader
- Web data scraper

[0.1.0]: https://github.com/vietnam-ai-challenge/kg-generator/releases/tag/v0.1.0
