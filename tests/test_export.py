"""Tests for the stable exported/uploaded node schema."""

from kg_generator.export.exporter import GraphExporter


def test_entity_export_has_exact_attribute_allowlist():
    exported = GraphExporter._normalize_node_props({
        "id": "entity:1",
        "name": "Alice",
        "type": "PERSON",
        "description": "A person",
        "importanceScore": 0.5,
        "confidenceScore": 0.9,
        "embedding": [0.1],
        "aliases": ["alice"],
        "source": ["chunk:1"],
        "updatedAt": "ignored",
    })

    assert set(exported) == {
        "id",
        "name",
        "type",
        "description",
        "importanceScore",
        "confidenceScore",
        "embedding",
    }


def test_document_export_has_exact_attribute_allowlist():
    exported = GraphExporter._normalize_node_props({
        "id": "document:1",
        "name": "article.txt",
        "type": "Document",
        "description": "Source document",
        "source": ["article.txt"],
        "chunk_count": 2,
        "aliases": ["ignored"],
    })

    assert set(exported) == {
        "id",
        "name",
        "type",
        "description",
        "source",
        "chunk_count",
    }


def test_chunk_export_has_exact_attribute_allowlist():
    exported = GraphExporter._normalize_node_props({
        "id": "chunk:1",
        "name": "ignored display name",
        "type": "Chunk",
        "source": ["article.txt"],
        "text": "Chunk text",
        "tokenCount": 2,
        "index": 0,
        "description": "ignored",
    })

    assert set(exported) == {
        "id",
        "type",
        "source",
        "text",
        "tokenCount",
        "index",
    }
