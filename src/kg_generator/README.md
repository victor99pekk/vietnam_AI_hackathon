# KG Generator Architecture

This package has two connected workflows.

```text
Raw documents → curate → curated dataset + audit
Curated documents → extract → resolve → graph → evaluate/export
```

## Curation workflow

`ingest/` loads files. `dedup/` profiles quality and finds repeated content. `curate/` performs language-aware normalization, layered duplicate decisions, sentence-safe token records, semantic review, deterministic shards, and provenance audits.

## Knowledge Graph workflow

`extract/` finds entities and relationships. `resolve/` merges references to the same entity. `graph/` creates the KG. `evaluate/` measures its quality, and `export/` writes files for downstream tools.

## Important distinction

- `dedup/` works on **documents** before extraction.
- `resolve/` works on **entities** after extraction.

Read the README in each component folder for its specific contract.
