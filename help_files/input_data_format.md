# Input Data Format Specification

All data entering the KG generation pipeline **must** follow the JSONL (JSON Lines) format defined below. This ensures consistent metadata handling across all pipeline stages (ingestion, curation, extraction, resolution, export).

---

## Required Format

Each line in a `.jsonl` file is a single, complete JSON object. Every object **must** include the following four fields:

| Field    | Type   | Required | Description |
|----------|--------|----------|-------------|
| `id`     | string | ✅ Yes   | Unique identifier for the document. Must be stable across pipeline runs (used for deduplication and version tracking). |
| `text`   | string | ✅ Yes   | Full body text of the document. Must be non-empty and UTF-8 encoded. |
| `title`  | string | ✅ Yes   | Human-readable title or heading of the document. Used for entity linking and graph node labels. |
| `url`    | string | ✅ Yes   | Source URL or canonical reference. Used for provenance tracking and metadata management (see [Problem Description — Component 3](Problem_description.md#3-labeling--metadata-management)). |

### Example

```jsonl
{"id": "12", "text": "Anarchism is a political philosophy and movement that is skeptical of all justifications for authority...", "title": "Anarchism", "url": "https://en.wikipedia.org/wiki/Anarchism"}
{"id": "39", "text": "Albedo is the fraction of sunlight that is diffusely reflected by a body...", "title": "Albedo", "url": "https://en.wikipedia.org/wiki/Albedo"}
```

---

## Field Semantics

### `id`
- **Purpose**: Stable document identity across re-ingestion runs.
- **Constraints**: String, unique within a dataset. Prefer numeric or slug-based IDs (e.g., Wikipedia page IDs, UUIDs, or `{source}_{counter}`).
- **Used by**: Deduplication, versioning (`replace old versions with new` for same `id`), export traceability.

### `text`
- **Purpose**: The raw document content fed into the extraction pipeline.
- **Constraints**: Non-empty, UTF-8, no length limit (chunking happens downstream in the ingestion stage).
- **Used by**: Entity extraction, relation extraction, chunking, QA pair generation.

### `title`
- **Purpose**: Document-level label for graph nodes and human-readable references.
- **Constraints**: Non-empty string. Should be the canonical title of the document.
- **Used by**: Graph node naming, entity resolution, QA context headers.

### `url`
- **Purpose**: Provenance link back to the original source.
- **Constraints**: Valid URL string. May be a file path for local documents (`file:///path/to/doc`) if no web source exists.
- **Used by**: Metadata management, source manifest generation, dataset auditing.

---

## Supported File Extensions

The pipeline loader recognizes the following extensions for JSONL input:

| Extension | Behavior |
|-----------|----------|
| `.jsonl`  | Parsed line-by-line as JSON objects with the schema above. |
| `.txt`    | **Legacy only** — entire file treated as one document with no metadata. **Not recommended** for new datasets. |

> **Note**: `.txt` support is retained for backward compatibility but will be deprecated. All new datasets should use `.jsonl`.

---

## Converting Existing `.txt` Files

If you have plain `.txt` files, convert them to `.jsonl` using the following template:

```python
import json
from pathlib import Path

txt_path = Path("data/debugg_sample/alan_turing.txt")
jsonl_path = txt_path.with_suffix(".jsonl")

record = {
    "id": txt_path.stem,                    # e.g., "alan_turing"
    "text": txt_path.read_text(encoding="utf-8"),
    "title": txt_path.stem.replace("_", " ").title(),  # e.g., "Alan Turing"
    "url": f"file://{txt_path.resolve()}",  # local file reference
}

jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
```

---

## Validation Checklist

Before running the pipeline, verify that every `.jsonl` file:

- [ ] Contains exactly one JSON object per line (no pretty-printing, no trailing commas)
- [ ] Every line has all four required fields (`id`, `text`, `title`, `url`)
- [ ] `text` is non-empty and UTF-8 encoded
- [ ] `id` values are unique across all files in the input directory
- [ ] `title` values are non-empty
- [ ] `url` values are valid strings (URL or `file://` URI)

---

## Related Documents

- [Problem Description — Component 3: Labeling & Metadata Management](Problem_description.md#3-labeling--metadata-management)
- [Wikipedia Dataset Documentation](wikipedia_dataset.md)
- [Pipeline Configuration Reference](../configs/debug.yaml)
