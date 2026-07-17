# Deduplication Experiments

The toolkit supports two document-level duplicate methods. Each run writes an immutable dataset version, a provenance manifest, `audit.csv`, and `duplicate_matches.csv` so experiments can be compared.

## 1. Surface-text baseline: MinHash

```bash
kg-gen curate -i raw_data/ -m manifest.yaml \
  --dedup-method minhash --dedup-threshold 0.85 \
  --experiment-id minhash-085
```

This combines exact SHA-256 hashes with MinHash candidates over character 3-grams. Candidates are checked with measured Jaccard overlap before they are marked as near duplicates. Use it for copied or lightly edited documents.

## 2. Semantic experiment: multilingual embeddings

```bash
uv pip install -e ".[embeddings]"
kg-gen curate -i raw_data/ -m manifest.yaml \
  --dedup-method semantic --dedup-threshold 0.92 \
  --experiment-id semantic-mpnet-092
```

This uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` and cosine similarity. It can find paraphrases, including cross-lingual matches, but it is more expensive and may remove useful related text. It compares every pair and is capped at 5,000 documents per run; it is an experiment-sized semantic baseline, not a web-scale ANN implementation of SemDeDup.

## Required review

For each run, review `duplicate_matches.csv` before accepting deletions. Record the method, threshold, model, dataset version, and manual judgement of a sample of matches. Use a new dataset version for every run.

## Current scope

Duplicate decisions are at the **document level**. Line and paragraph duplicate detection are not yet implemented; they should be logged separately before any workflow rewrites document content.
