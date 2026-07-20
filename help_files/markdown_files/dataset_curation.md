# Dataset Curation Toolkit

`kg-gen curate` converts mixed text files into an immutable, auditable LLM-ready corpus. Each run declares one language (`en` or `vi`) in its source manifest. English uses spaCy sentence rules; Vietnamese uses underthesea word and sentence segmentation.

## Run it

```bash
kg-gen curate \
  -i data/sample/ \
  -m configs/example_source_manifest.yaml \
  -o output/curated_datasets \
  --device cuda
```

Install curation dependencies first:

```bash
uv pip install -e ".[curation,dev]"
```

`faiss-cpu` is included for portable runs. On cloud CUDA images, install the FAISS GPU build matched to the image's CUDA version; the pipeline uses it automatically when available.

Run the real BGE-M3 CUDA integration check only on a prepared GPU worker:

```bash
RUN_CURATION_GPU_TEST=1 python -m pytest tests/test_curation.py -m integration
```

The output path is immutable: `<output>/<dataset_name>/<version>/`. Choose a new version in the source manifest for a new run. This prevents accidental overwrites.

## Source manifest

Provide YAML or JSON with the following required fields:

```yaml
dataset_name: my-legal-corpus
version: v1
license: CC-BY-4.0
source: https://example.org/open-data
language: en  # en or vi
collection_date: 2026-07-17  # optional
```

The tool records these values but does not infer or validate licensing. Curate only material you are allowed to use.

## Curation flow

1. Normalize Unicode to NFC, remove invalid controls, repair safe mojibake, and preserve paragraph breaks, punctuation, casing, and Vietnamese diacritics.
2. Reject only empty/corrupt or below-minimum documents. Symbol density, repeated lines, and repeated characters remain review flags.
3. Auto-reject exact hashes and strong document-level MinHash matches over language-aware word 5-grams (`--surface-threshold`, default `0.90`).
4. Split accepted documents into sentence-aligned records of at most 2,048 BGE-M3 tokenizer tokens. Text is never truncated; every record links to its parent document and character span.
5. Embed accepted records in token-bounded CUDA batches. BGE-M3 cosine neighbors above `0.92` become review-only semantic duplicate candidates; they are never deleted automatically.
6. Write canonical JSONL plus deterministic, hash-shuffled token-budget shards for training.

Use `--no-semantic-review` only for a surface-dedup experiment. Use `--resume` after an interrupted run with identical options and source manifest; it reuses staged embedding checkpoints.

## Artifacts

- `curated.jsonl` contains accepted records in stable parent-document order. Unsplit records retain their source ID; split IDs use `:part-N`.
- `shards/batch-*.jsonl` contain identical accepted records in deterministic hash-shuffled order, bounded by one million model tokens by default.
- `audit.csv` contains every source document, quality signals, automatic decision, duplicate evidence, and accepted-record count.
- `record_audit.csv` contains every derived record, parent provenance, character span, model token count, review flags, and decision.
- `duplicate_matches.csv` records direct matches with `scope` and `decision`; semantic rows are `review_only`.
- `batch_manifest.json`, `quality_report.json`, and `dataset_manifest.json` record shard hashes, diagnostics, provenance, settings, model revision, and artifact hashes.

## Standard operating procedure

1. Confirm that every source is legally reusable; create one source manifest for the intended dataset version.
2. Run `kg-gen curate` on all input sources together, so duplicates are detected across sources.
3. Review `audit.csv`, `record_audit.csv`, and every `semantic_duplicate_candidate` before publishing. Semantic candidates are retained until a later, explicit policy changes them.
4. Review `quality_report.json` and `batch_manifest.json` for source imbalance, high duplicate rate, missing content, or unexpected token sizes.
5. Preserve the complete version directory. Create a new manifest version rather than editing published artifacts.

The quality values are explainable triage signals, not downstream-model guarantees. PII redaction, toxicity filtering, mixed-language auto-detection, and KG chunking are intentionally out of scope for this release.
