# Graph Construction

## Purpose

Build a structured Knowledge Graph from resolved entities and extracted relationships.

## Input → output

Resolved entities and triples → a directed graph with typed nodes and edges.

## Key files

- `builder.py` creates and validates the graph against the configured ontology.
- `enrich.py` is the future extension point for linking entities to external knowledge bases.

## Place in pipeline

After entity resolution; before KG evaluation and export.
