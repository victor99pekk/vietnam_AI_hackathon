# Knowledge Graph Generator — Usage Guide

## Quick Start

```bash
# Install
pip install -e .

# Download spaCy model (for English extraction)
python -m spacy download en_core_web_lg

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
5. **Build Graph** — Construct directed graph, validate against ontology
6. **Export** — JSON, GraphML, Neo4j CSV, RDF/Turtle, Cytoscape.js

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
pip install -e ".[vi]"
kg-gen run -l vi -i vietnamese_data/
```

### Using LLM-based Extraction

```bash
pip install -e ".[llm]"
export OPENAI_API_KEY=your_key
kg-gen run --llm -i data/
```

### Neo4j Export

```bash
pip install -e ".[neo4j]"
# Update config: graph_backend: neo4j
kg-gen run -c configs/pipeline.yaml
```
