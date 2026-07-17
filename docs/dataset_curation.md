# Dataset Curation Toolkit

The curation command converts mixed text files into a clean, auditable dataset for preliminary LLM data evaluation. It is designed for English-first development and is Unicode-safe for a later Vietnamese corpus.

## Run it

```bash
kg-gen curate \
  -i data/sample/ \
  -m configs/example_source_manifest.yaml \
  -o output/curated_datasets
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

## Artifacts

- `curated.jsonl` contains accepted documents, source metadata, content hashes, and quality scores.
- `audit.csv` contains every input record, its quality signals, decision, rejection reasons, and duplicate provenance.
- `quality_report.json` summarizes preliminary completeness, format/missing-content errors, duplicate rate, source composition, and language-agnostic diversity diagnostics.
- `dataset_manifest.json` records provenance, input and artifact hashes, settings, timestamps, counts, and a deterministic configuration hash.

## Standard operating procedure

1. Confirm that every source is legally reusable; create one source manifest for the intended dataset version.
2. Run `kg-gen curate` on all input sources together, so duplicates are detected across sources.
3. Review `audit.csv`; inspect low-scoring documents and all `near_duplicate` decisions before publishing the curated dataset.
4. Review `quality_report.json` for unexpected source imbalance, high duplicate rate, missing content, or unusually low n-gram diversity.
5. Preserve the complete version directory. Create a new manifest version rather than editing published artifacts.

The current quality and diversity values are explainable triage signals, not a guarantee of downstream model performance. Toxicity filtering, Vietnamese tokenization, semantic embeddings, and LLM-as-judge scoring are planned extensions.
