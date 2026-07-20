# Deduplication Experiments

Every curate run uses layered duplicate handling. Exact hashes and strong surface duplicates are automatic; semantic duplicates are review candidates only. Each run writes an immutable dataset version, provenance manifest, source/record audits, and match evidence.

## Surface-text decision: MinHash

```bash
kg-gen curate -i raw_data/ -m manifest.yaml \
  --surface-threshold 0.90 --no-semantic-review
```

This combines exact SHA-256 hashes with MinHash candidates over language-aware word 5-grams. Candidates are checked with measured Jaccard overlap before automatic deletion. Use it for copied or lightly edited documents.

## Semantic review: multilingual BGE-M3

```bash
uv pip install -e ".[curation]"
kg-gen curate -i raw_data/ -m manifest.yaml \
  --semantic-model BAAI/bge-m3 \
  --semantic-review-threshold 0.92 --device cuda
```

This embeds token-limited records with BGE-M3 and searches the 20 nearest neighbors through FAISS. It finds likely paraphrases, but never removes them: matched records gain `semantic_duplicate_candidate` in `record_audit.csv`, and match rows have `decision=review_only`.

## Required review

For each run, review every automatic deletion and semantic candidate. Record the thresholds, model revision, dataset version, and manual judgement. Use a new manifest version for every published run.

## Current scope

Surface decisions occur at the document level. Exact record duplicates are also removed after sentence-safe splitting; semantic record matches remain review-only. PII, toxicity, and mixed-language detection are not included.
