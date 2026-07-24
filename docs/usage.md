# Knowledge Graph Generator — Usage Guide

## Quick Start

```bash
# Using uv
uv venv
source .venv/bin/activate
uv pip install -e "."
python -m spacy download en_core_web_sm

# Run with defaults on sample data
kg-gen quick -i data/sample/

# Run with config
kg-gen run -c configs/pipelines/default.yaml
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
| `--chunk-method` | `none`, `fixed`, `sentence`, or `semantic` |
| `--quality-method` | `none` or `heuristic` |
| `--document-dedup-method` | Document-level dedup strategy |
| `--dedup-method` | Chunk-level dedup strategy |
| `--resolve-method` | `string` or multilingual `embedding` |

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

The manifest declares `en` or `vi`. Curate applies exact and MinHash deletion, BGE-M3 semantic review, sentence-safe 2,048-token records, and deterministic training shards. See `src/kg_generator/curate/README.md` for details.

### `kg-gen scrape`

Collect robots-aware web pages into pipeline-ready JSONL with a provenance
manifest and audit CSV:

```bash
uv sync --extra scraper
kg-gen scrape --seed-file data/download_data/seeds/vietnamese_sources.txt \
  --language vi --max-pages 50 -o data/scraped/vietnamese_demo
```

The KG loader retains scraper title, URL, license, domain, collection time,
crawler identity, content hash, and inferred type on source graph nodes.

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

See `configs/pipelines/default.yaml` for all options and `configs/default_ontology.yaml` for entity/relation schema.

Strategy configuration supports nested YAML while retaining the old flat keys:

```yaml
pipeline:
  language: vi
  chunking:
    method: sentence          # none | fixed | sentence | semantic
    target_tokens: 450
    overlap_tokens: 60
  quality:
    method: heuristic
  deduplication:
    document_method: minhash  # none | exact | minhash | simhash | ngram | semantic | layered
    chunk_method: semantic
    semantic_threshold: 0.92
  extraction:
    method: graphgen          # offline | graphgen
  resolution:
    method: embedding         # string | embedding
    threshold: 0.88
```

Every JSON export and `metrics.json` records the selected strategies and the
document/chunk counts before and after quality filtering and deduplication.

## Extending

### Adding Vietnamese Support

```bash
uv sync --extra vi --extra llm

# GraphGen quality path (does not require Underthesea at runtime)
kg-gen run -l vi --llm -i vietnamese_data/

# Offline path (requires Underthesea)
kg-gen run -l vi --no-llm -i vietnamese_data/

# Ready-to-run example configuration
kg-gen run -c configs/pipelines/vietnamese.yaml

# Fast, dependency-light alternative
kg-gen run -c configs/pipelines/vietnamese_fast.yaml
```

Vietnamese names, descriptions, evidence, and source text remain Vietnamese.
Entity types and structural predicates remain the same English schema identifiers
used by English graphs. If Underthesea is missing, the offline path stops with an
install command instead of exporting an empty graph.

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
# Update configs/pipelines/default.yaml: graph_backend: neo4j
kg-gen run -c configs/pipelines/default.yaml
```
