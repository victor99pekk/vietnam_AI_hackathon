# Extraction

## Purpose

Turn document text into entities and relationships that can become graph nodes and edges.

## Input → output

Clean chunks → entity records and descriptive relationship records containing
`source`, `target`, `description`, and source-chunk provenance.

## Key files

- `entities.py` extracts named entities using English or Vietnamese-capable backends.
- `relations.py` provides the local rule-based baseline.
- `graphgen_prompts.py` contains the Figure 8 extraction and Figure 9
  description-aggregation prompts.
- `graphgen.py` uses DeepSeek to run those prompts, including GraphGen's
  iterative missed-entity/relationship gleaning. Run it with `kg-gen run --llm`.

DeepSeek extraction uses `deepseek-v4-pro` by default. It preserves entity names in
their original Unicode script, so the same path supports English and Vietnamese.
Knowledge relationships do not receive predicted predicate types: `RELATION` is only
the neutral storage label, while meaning is carried by the relationship description.
Repeated entity and relationship descriptions are merged with the Figure 9 prompt.

## Place in pipeline

After document curation or quality/deduplication; before entity resolution and graph construction.
