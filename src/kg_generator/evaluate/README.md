# KG Evaluation

## Purpose

Measure the preliminary quality of the completed Knowledge Graph.

## Input → output

Graph, entities, and triples → a metrics dictionary.

## Key files

- `metrics.py` calculates completeness, consistency, duplication, missing information, format errors, labeling quality, and reusability.
- `evaluation/graphgen/` is the separate GraphGen-style experiment for audited
  k-hop subgraphs and multi-hop SFT QA generation.

## Place in pipeline

After graph construction. These metrics assess the KG, while `curate/` reports document-dataset quality.
