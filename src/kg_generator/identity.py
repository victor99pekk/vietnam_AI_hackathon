"""Deterministic identifiers shared by extraction, graph, and export stages."""

from __future__ import annotations

import hashlib
import json
import unicodedata


def _canonical_part(value: object) -> str:
    """Return a Unicode-safe, whitespace-normalized representation for hashing."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def stable_id(kind: str, *parts: object) -> str:
    """Create a deterministic, namespaced ID from one or more identity parts."""
    normalized_kind = _canonical_part(kind).replace(" ", "_") or "node"
    payload = json.dumps(
        [_canonical_part(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{normalized_kind}:{digest}"


def entity_id(label: str, name: str) -> str:
    """Stable identity for an extracted entity."""
    return stable_id("entity", label, name)


def document_id(source: str, source_document_id: str = "") -> str:
    """Stable identity for a source document or source record."""
    return stable_id("document", source, source_document_id or source)


def chunk_id(parent_id: str, index: int, text: str) -> str:
    """Stable identity for a particular chunk within a document."""
    return stable_id("chunk", parent_id, index, text)
