# Vietnamese Global Knowledge Graph Runbook

This is the operator handoff for building a large Vietnamese graph with the
highest quality that the current repository supports at scale.

## 1. Important limits

- Direct Neo4j is the scale-safe graph backend, but it supports only string
  entity resolution.
- In-memory embedding resolution is approximately quadratic in entity count.
- Semantic deduplication is all-pairs and capped at 5,000 records.
- GraphGen requests are sequential. With three gleanings, one chunk can make
  as many as seven extraction requests, plus possible summary requests.
- Document IDs include both the record ID and input file path. Batch paths and
  membership must therefore remain stable: do not rename, move, or reshuffle an
  existing record into a different batch after it has been loaded.
- MinHash deduplication is global only within one pipeline invocation. Dedup the
  full source collection before splitting it into extraction batches whenever
  memory permits.
- The heuristic quality filter rejects empty/corrupt and very short content.
  Repeated lines, excessive symbols, and repeated characters are review flags,
  not automatic rejections.

## 2. Required input schema

Use UTF-8 JSONL with one object per source record. Keep metadata flat and use a
permanent canonical URL or stable ID.

```json
{"id":"permanent-source-id","url":"https://example.vn/permanent-url","title":"Vietnamese title","text":"Vietnamese source text...","license":"CC-BY-4.0","source_domain":"example.vn","scraped_at":"2026-07-18T00:00:00Z","crawler":"collector-name/version","content_hash":"sha256-value","inferred_type":"news_article"}
```

Rules:

1. `text` must contain the complete source text.
2. `url` is preferred as the source record ID; otherwise `id` is used.
3. Never recycle an ID or URL for a different source.
4. Keep Vietnamese diacritics and NFC Unicode.
5. Keep title, URL, license, domain, timestamp, crawler, content hash, and
   inferred type at the top level. Do not hide them in a nested metadata object.
6. Do not put the same record into multiple extraction batches.
7. Once a batch has been ingested, never rename or move that batch file.

## 3. Install dependencies

```bash
uv sync \
  --extra curation \
  --extra embeddings \
  --extra llm \
  --extra neo4j \
  --extra vi \
  --extra mongo
```

MongoDB is optional. Omit `--extra mongo` when raw-document version archiving is
not required.

## 4. Configure secrets

```bash
export NEO4J_URI='neo4j+s://REPLACE_ME'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='REPLACE_ME'
export DEEPSEEK_API_KEY='REPLACE_ME'
```

Optional versioned document archive:

```bash
export MONGO_URI='mongodb://REPLACE_ME'
export MONGO_DATABASE='kg_documents'
```

Do not store secrets in YAML or commit them to Git.

## 5. Curate and quality-check the corpus

Copy and complete the source manifest:

```bash
cp configs/vietnamese_global_source_manifest.example.yaml \
  configs/vietnamese_global_source_manifest.yaml
```

Run curation over all source directories together when feasible:

```bash
kg-gen curate \
  -i data/raw_vietnamese/ \
  -m configs/vietnamese_global_source_manifest.yaml \
  -o output/curated_datasets \
  --surface-threshold 0.90 \
  --semantic-review-threshold 0.94 \
  --semantic-model BAAI/bge-m3 \
  --device cuda \
  --max-record-tokens 2048 \
  --embedding-batch-tokens 16384 \
  --shard-tokens 1000000
```

If interrupted, rerun the identical command with `--resume`. Do not change any
option when resuming.

Before graph extraction, inspect:

- `audit.csv`: source-document accept/reject decisions.
- `record_audit.csv`: derived records, spans, and review flags.
- `duplicate_matches.csv`: exact, MinHash, and semantic candidates.
- `quality_report.json`: counts, duplicate rate, source balance, token sizes.
- `dataset_manifest.json`: configuration and artifact hashes.

Semantic matches are review-only. They are not automatically removed. Review a
sample, especially across different publishers covering the same event: similar
articles may contain distinct facts and should not be deleted automatically.

The curation output nests some original metadata. Before extraction, construct
flat immutable batch JSONL files matching the schema in section 2 so provenance
appears correctly on Document nodes.

## 6. Make stable extraction batches

Size batches by resulting chunk count, not just document count.

Recommended operational target:

- 500-2,000 expected chunks per batch for maximum-quality GraphGen.
- Approximately 100-1,000 documents depending on document length.
- One batch writer at a time unless concurrent Neo4j resolution has been tested.
- Partition by immutable source/date boundaries, never by random reshuffling.

Store batches permanently, for example:

```text
data/global_vi_batches/
  batch-00000.jsonl
  batch-00001.jsonl
  batch-00002.jsonl
```

Maintain a batch ledger containing path, SHA-256, record count, expected chunk
count, start time, completion time, and run result. If a previously loaded source
must be corrected, archive the old batch and hash, update that source in the same
batch path, and rerun the entire batch. Moving the correction to a new file path
will produce a different Document ID even when its URL or record ID is unchanged.

## 7. Validate a pilot before the global run

First run 100-500 representative Vietnamese documents covering several source
types. Manually label at least:

- 100 entity pairs for merge/not-merge threshold calibration.
- 100 extracted relationships for endpoint and grounding accuracy.
- 50 chunk boundaries for coherence.
- 50 dedup pairs near each configured threshold.

Tune only after examining false positives and false negatives. False entity
merges are more damaging than duplicate nodes, so prioritize merge precision.

## 8. Build the graph

Configuration:

```text
configs/pipelines/vietnamese_global_quality.yaml
```

Initial build only:

```bash
kg-gen run \
  -c configs/pipelines/vietnamese_global_quality.yaml \
  -i data/global_vi_batches/batch-00000.jsonl \
  -o output/global-vi/batch-00000 \
  --clear
```

Every later batch, without `--clear`:

```bash
kg-gen run \
  -c configs/pipelines/vietnamese_global_quality.yaml \
  -i data/global_vi_batches/batch-00001.jsonl \
  -o output/global-vi/batch-00001
```

Repeat sequentially for every immutable batch. A successful direct Neo4j run
writes batch metrics to `OUTPUT_DIR/metrics.json`.

Never use `--clear` for an incremental batch. It deletes the global graph.

## 9. Required graph checks after every batch

Record the processing counts from `metrics.json`:

- loaded and cleaned documents;
- documents after quality filtering and deduplication;
- chunks created and retained;
- extracted entities and triples;
- final Neo4j node and edge counts.

Stop and investigate when:

- more than 20% of documents disappear unexpectedly;
- chunk count per document changes sharply from previous batches;
- extracted entity or relationship count is near zero;
- API error rate increases;
- generic concepts dominate named entities;
- a source produces an abnormally dense graph;
- duplicate names grow rapidly across batches.

Also sample source-backed edges and verify that every relationship description is
entailed by its source chunk.

## 10. Vietnamese QA generation

QA is a separate downstream pipeline. The commands below require a faithful
`knowledge_graph.json` containing entity names, relationship descriptions, and
source chunk IDs.

**Current blocker:** `kg-gen neo4j-download` currently omits entity display names
from Entity nodes and omits relationship descriptions, evidence, and source chunk
IDs. Its output is adequate for structural inspection but not for grounded
GraphGen QA. Do not generate production QA from that download until the downloader
is updated to preserve those properties.

After that exporter issue is fixed, first sample subgraphs without an API call:

```bash
uv run python evaluation/run_eval.py \
  --method graphgen \
  --sample-only \
  --kg PATH_TO_EXPORTED_KNOWLEDGE_GRAPH.json \
  -c configs/evaluation/vietnamese_global_eval.yaml \
  -o output_eval/global-vi
```

Review `subgraphs.jsonl` and `sampling_audit.jsonl`. Then generate QA:

```bash
uv run python evaluation/run_eval.py \
  --method graphgen \
  --kg PATH_TO_EXPORTED_KNOWLEDGE_GRAPH.json \
  -c configs/evaluation/vietnamese_global_eval.yaml \
  -o output_eval/global-vi
```

Review `qa.jsonl` and `qa_audit.jsonl`. The automatic validator confirms that at
least two known connected edges were cited, but it does not prove semantic
faithfulness. Human review or an independent Vietnamese LLM judge is still
required before training.

The direct Neo4j build does not create `knowledge_graph.json`. The current export
command is:

```bash
kg-gen neo4j-download -o output/global-vi/knowledge_graph.json
```

Verify the downloaded JSON before QA. Domain Entity nodes must have readable
Vietnamese `name` values, and `RELATION` triples must have non-empty Vietnamese
`description` plus `source_chunk_id` or `source_chunk_ids`. If those fields are
missing, stop; the QA generator will not have grounded premises.

## 11. Current quality ceiling

These problems cannot be solved by YAML alone:

1. Direct Neo4j resolution searches only same-type substring-compatible names.
   Acronyms and many Vietnamese aliases remain separate.
2. Direct Neo4j does not perform GraphGen cross-document description aggregation.
3. Existing entity descriptions are not comprehensively merged when new sources
   add information.
4. The ontology is not automatically loaded or enforced by normal pipeline YAML.
5. Quality review flags do not automatically remove web boilerplate.
6. Deduplication is not cross-batch after extraction batching.
7. GraphGen extraction is sequential and lacks resumable per-chunk checkpoints.
8. The current Neo4j downloader loses semantic relationship and provenance fields
   needed by the GraphGen QA generator.

For a genuinely global production graph, plan a second-stage entity-resolution
and description-aggregation job in Neo4j. Treat the configuration above as the
highest-quality scale-safe first pass supported by the current code.
