# Export

## Purpose

Save the completed Knowledge Graph in formats that other tools can use.

## Input → output

Graph, entities, and triples → JSON, GraphML, Neo4j CSV, RDF/Turtle, or Cytoscape.js files.

## Key files

- `exporter.py` writes each supported output format.
- `neo4j_upload.py` provides the programmatic Neo4j uploader.

Neo4j nodes and relationships are matched by stable IDs. `kg-gen neo4j-upload`
replaces existing documents with matching IDs, including their old chunks and
source-backed relationships. For relationships shared with other documents,
only the replaced chunk IDs are removed; the relationship is deleted when no
sources remain. Unrelated documents remain. Pass `--clear` only when a full
database replacement is intended.

Incremental uploads, including `make add`, merge relationship source chunk IDs
and evidence instead of overwriting provenance already stored in Neo4j.

GraphGen knowledge edges are uploaded as `:RELATION` with `description` and
`sourceChunkIds`. Structural edges retain `:MENTIONS`, `:PART_OF`, and `:NEXT`.

## Place in pipeline

Final KG step, after construction and evaluation.
