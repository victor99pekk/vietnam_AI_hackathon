# GraphGen-style QA generation

## Purpose

Turn the descriptive `RELATION` edges produced by the KG pipeline into bounded,
auditable subgraphs and then into multi-hop SFT question/answer records.

This is a new experiment. The older `data_eval` and `model_eval` generators are
left unchanged so their outputs can be compared.

## Process

1. Read entity nodes and descriptive knowledge edges from
   `knowledge_graph.json`.
2. Start one candidate subgraph from each knowledge edge.
3. Expand through connected edges in both directions, up to `max_depth`.
4. Stop at `max_extra_edges` or the estimated premise-token budget.
5. Rank by comprehension loss when it exists. Until Step 2 of GraphGen is
   implemented, use a deterministic stable-ID fallback and record that fact.
6. Give accepted subgraphs to DeepSeek and require a question that uses an
   ordered connected path of at least two known edges.

The graph traversal is deterministic code. Only Step 6 calls an LLM.

## Artifacts

- `subgraphs.jsonl`: selected nodes, edges, token estimate and source chunks.
- `sampling_audit.jsonl`: one include/reject decision per considered edge.
- `qa.jsonl`: accepted SFT instruction/response records and evidence chains.
- `qa_audit.jsonl`: rejected, failed and accepted generation attempts.
- `results.json`: counts, settings summary and current paper-faithfulness limits.

## Run

Inspect the subgraphs without an API call:

```bash
uv run python evaluation/run_eval.py \
  --method graphgen \
  --sample-only \
  --kg generated_KGs/output_debug/knowledge_graph.json
```

Generate multi-hop QA with the configured DeepSeek model:

```bash
uv run python evaluation/run_eval.py \
  --method graphgen \
  --kg generated_KGs/output_debug/knowledge_graph.json
```

Configuration is under `graphgen` in `evaluation/eval_config.yaml`.

## Current boundary

Implemented from the GraphGen organization/generation approach: edge-seeded
k-hop expansion, bidirectional traversal, premise-length control, configurable
edge ranking, provenance, and multi-hop QA generation.

Not yet implemented: GraphGen's trainee-model comprehension assessment and ECE
loss calculation. Therefore `max_loss` currently falls back to stable edge IDs
unless real `comprehension_loss` values are supplied.
