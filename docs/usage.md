# Knowledge Graph Generator — Usage Guide

## Quick Start

```bash
# Using uv (recommended — see docs/UV_SETUP.md if you're new to uv)
uv venv
source .venv/bin/activate
uv pip install -e "."
python -m spacy download en_core_web_sm

# Run with defaults on sample data
kg-gen quick -i data/sample/

# Run with config
kg-gen run -c configs/pipeline.yaml
```

## CLI Commands

### `kg-gen run`

Full pipeline with configuration.

| Option | Description |
|---|---|
| `-i, --input` | Input file/directory (repeatable) |
| `-c, --config` | Path to YAML config file |
| `-o, --output` | Output directory (default: `./output`) |
| `-l, --language` | `en` or `vi` (default: `en`) |
| `--llm` / `--no-llm` | Enable LLM-based extraction |

### `kg-gen quick`

Run with sensible defaults, no config needed.

```bash
kg-gen quick -i my_data.txt -o ./results
```

### `kg-gen curate`

Build an immutable, audited corpus before LLM training or evaluation.

```bash
uv pip install -e ".[curation]"
kg-gen curate -i raw_data/ -m manifest.yaml -o output/curated_datasets --device cuda
```

The manifest declares `en` or `vi`. Curate applies exact and MinHash deletion, BGE-M3 semantic review, sentence-safe 2,048-token records, and deterministic training shards. See [dataset curation](dataset_curation.md) for options and review procedure.

### `kg-gen evaluate`

Run quality evaluation on an existing KG.

```bash
kg-gen evaluate -i output/knowledge_graph.json
```

## Pipeline Stages

1. **Ingest** — Load from TXT, JSON, CSV, JSONL; normalize text
2. **Dedup & Quality Filter** — Remove low-quality and duplicate content
3. **Extract** — NER + relation extraction → (subject, predicate, object) triples
4. **Resolve** — Merge duplicate entity mentions
5. **Build Graph & Export** — Construct directed graph, validate against ontology, export to multiple formats

## Output Files

| File | Description |
|---|---|
| `knowledge_graph.json` | Full KG with nodes, edges, entities, triples |
| `knowledge_graph.graphml` | GraphML for Gephi / Cytoscape |
| `neo4j_import/` | CSV files for Neo4j bulk import |
| `metrics.json` | Quality evaluation metrics |

## Configuration

See `configs/pipeline.yaml` for all options and `configs/default_ontology.yaml` for entity/relation schema.

## Extending

### Adding Vietnamese Support

```bash
uv pip install -e ".[vi]"
kg-gen run -l vi -i vietnamese_data/
```

### Using Embedding-based Entity Resolution

```bash
uv pip install -e ".[embeddings]"
# The resolver will automatically use sentence-transformers for better accuracy
```

### Using LLM-based Extraction

```bash
uv pip install -e ".[llm]"
export OPENAI_API_KEY=your_key
kg-gen run --llm -i data/
```

### Neo4j Export

```bash
uv pip install -e ".[neo4j]"
# Update configs/pipeline.yaml: graph_backend: neo4j
kg-gen run -c configs/pipeline.yaml
```
