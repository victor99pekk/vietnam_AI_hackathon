"""Tests for graph construction."""

import networkx as nx

from kg_generator.graph.builder import GraphBuilder


def test_build_graph():
    entities = [
        {"name": "Alice", "label": "PERSON", "mentions": ["Alice"], "confidence": 0.9},
        {"name": "Acme Corp", "label": "ORG", "mentions": ["Acme Corp"], "confidence": 0.95},
    ]
    triples = [("Alice", "works_at", "Acme Corp")]

    builder = GraphBuilder()
    graph = builder.build(entities, triples)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.has_edge("Alice", "Acme Corp")
    assert "works_at" in graph.edges["Alice", "Acme Corp"]["predicates"]


def test_deduplication_removes_exact_duplicates():
    from kg_generator.dedup.near_dedup import Deduplicator
    from kg_generator.ingest.loader import Document

    docs = [
        Document(content="Unique document one.", doc_id="1"),
        Document(content="Unique document two.", doc_id="2"),
        Document(content="Unique document one.", doc_id="3"),  # exact dup of doc 1
    ]

    dedup = Deduplicator(method="ngram", threshold=0.9)
    result = dedup.deduplicate(docs)

    assert len(result) == 2
    ids = {d.doc_id for d in result}
    assert ids == {"1", "2"} or ids == {"2", "3"}


def test_quality_filter_removes_short_docs():
    from kg_generator.dedup.quality import QualityFilter
    from kg_generator.ingest.loader import Document

    docs = [
        Document(content="Short."),
        Document(content="A properly sized document with enough words to pass the quality filter check."),
    ]

    qf = QualityFilter(min_chars=40, min_words=5)
    result = qf.filter(docs)
    assert len(result) == 1
