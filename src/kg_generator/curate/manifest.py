"""Source-manifest validation and deterministic hashing helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SourceManifest:
    """User-supplied ownership and provenance information for a dataset."""

    dataset_name: str
    version: str
    license: str
    source: str
    language: str = "en"
    collection_date: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> "SourceManifest":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("Source manifest must contain a mapping.")
        required = ("dataset_name", "version", "license", "source")
        missing = [field for field in required if not str(data.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Source manifest is missing required field(s): {', '.join(missing)}")
        language = str(data.get("language", "en"))
        if language not in {"en", "vi"}:
            raise ValueError("Source manifest language must be 'en' or 'vi'.")
        return cls(
            dataset_name=str(data["dataset_name"]),
            version=str(data["version"]),
            license=str(data["license"]),
            source=str(data["source"]),
            language=language,
            collection_date=str(data["collection_date"]) if data.get("collection_date") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(serialized)
