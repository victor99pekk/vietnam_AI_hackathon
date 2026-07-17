# Graph Construction

## Purpose

Build a structured Knowledge Graph from resolved entities and extracted relationships.

## Input → output

Resolved entities and triples → a directed graph with typed nodes and edges.
Stable IDs are graph keys; names are display properties and may be duplicated.

In GraphGen mode, knowledge edges use the neutral `RELATION` storage label and
carry their meaning in a natural-language `description`. `MENTIONS`, `PART_OF`,
and `NEXT` are pipeline-structure edges rather than extracted knowledge predicates.

## Key files

- `builder.py` creates and validates the graph against the configured ontology.
- `enrich.py` is the future extension point for linking entities to external knowledge bases.
- `../identity.py` creates deterministic Document, Chunk, and Entity IDs.

## Place in pipeline

After entity resolution; before KG evaluation and export.
