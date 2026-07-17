# Wikimedia Wikipedia Pilot Dataset

Use this public dataset for a small, reproducible pipeline experiment before downloading a full language subset.

The dataset provides `id`, `url`, `title`, and cleaned article `text`, which map directly to the curation input schema. It is licensed under CC BY-SA 3.0 and GFDL; preserve attribution and share-alike requirements when using derived data.

## Install the downloader dependency

```bash
pip install -e ".[data]"
```

## Download a pilot sample

```bash
python scripts/download_wikipedia_sample.py \
  --language en --count 1000 \
  --output data/external/wikipedia_en_1000.jsonl

python scripts/download_wikipedia_sample.py \
  --language vi --count 1000 \
  --output data/external/wikipedia_vi_1000.jsonl
```

The script streams only the requested number of articles from `20231101.en` or `20231101.vi`. It writes a source manifest next to each JSONL file.

## Run curation

```bash
kg-gen curate \
  -i data/external/wikipedia_en_1000.jsonl \
  -m data/external/wikipedia_en_1000_manifest.yaml \
  --experiment-id wikipedia-en-minhash-085
```

Use a new manifest version for every experiment. Do not commit downloaded dataset files; they may be large and are reproducible from this script.
