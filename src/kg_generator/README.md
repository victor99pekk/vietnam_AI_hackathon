# KG Generator Architecture

This package has two connected workflows.

```text
Raw documents → curate → curated dataset + audit
Curated documents → extract → resolve → graph → evaluate/export
```

## Curation workflow

`ingest/` loads and normalizes files. `dedup/` profiles document quality and finds repeated documents. `curate/` saves accepted documents, audit decisions, quality reports, and source metadata.

## Knowledge Graph workflow

`extract/` finds entities and relationships. `resolve/` merges references to the same entity. `graph/` creates the KG. `evaluate/` measures its quality, and `export/` writes files for downstream tools.

## Important distinction

- `dedup/` works on **documents** before extraction.
- `resolve/` works on **entities** after extraction.

Read the README in each component folder for its specific contract.
