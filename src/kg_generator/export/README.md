# Export

## Purpose

Save the completed Knowledge Graph in formats that other tools can use.

## Input → output

Graph, entities, and triples → JSON, GraphML, Neo4j CSV, RDF/Turtle, or Cytoscape.js files.

## Key files

- `exporter.py` writes each supported output format.

## Place in pipeline

Final KG step, after construction and evaluation.
