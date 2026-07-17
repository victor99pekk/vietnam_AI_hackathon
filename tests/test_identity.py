"""Tests for deterministic graph identifiers."""

from kg_generator.identity import chunk_id, document_id, entity_id


def test_entity_id_is_stable_and_unicode_safe():
    first = entity_id("PERSON", "Nguyễn Trãi")
    second = entity_id(" person ", "  NGUYỄN   TRÃI ")

    assert first == second
    assert first.startswith("entity:")


def test_node_kinds_have_separate_namespaces():
    doc = document_id("data/article.txt", "article-1")
    chunk = chunk_id(doc, 0, "The first chunk")

    assert doc.startswith("document:")
    assert chunk.startswith("chunk:")
    assert doc != chunk


def test_chunk_id_changes_when_chunk_identity_changes():
    doc = document_id("data/article.txt", "article-1")

    assert chunk_id(doc, 0, "alpha") == chunk_id(doc, 0, "alpha")
    assert chunk_id(doc, 0, "alpha") != chunk_id(doc, 1, "alpha")
    assert chunk_id(doc, 0, "alpha") != chunk_id(doc, 0, "beta")
