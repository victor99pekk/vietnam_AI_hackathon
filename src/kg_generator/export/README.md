# Export

## Purpose

Save the completed Knowledge Graph in formats that other tools can use.

## Input → output

Graph, entities, and triples → JSON, GraphML, Neo4j CSV, RDF/Turtle, or Cytoscape.js files.

## Key files

- `exporter.py` writes each supported output format.
- `neo4j_upload.py` provides the programmatic Neo4j uploader.

Neo4j nodes and relationships are matched by stable IDs. `kg-gen neo4j-upload`
preserves existing data by default; pass `--clear` only when a full database
replacement is intended.

GraphGen knowledge edges are uploaded as `:RELATION` with `description` and
`sourceChunkIds`. Structural edges retain `:MENTIONS`, `:PART_OF`, and `:NEXT`.

## Place in pipeline

Final KG step, after construction and evaluation.
