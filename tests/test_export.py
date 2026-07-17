"""Tests for the stable exported/uploaded node schema."""

import json
import sys
from types import SimpleNamespace

from click.testing import CliRunner

from kg_generator import cli
from kg_generator.export.exporter import GraphExporter
from kg_generator.export.neo4j_upload import replace_documents


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


def test_replacing_document_prunes_shared_sources_and_deletes_unsupported_data():
    class FakeResult:
        def single(self):
            return {
                "chunk_ids": ["chunk:old"],
                "entity_ids": ["entity:old"],
            }

    class FakeSession:
        def __init__(self):
            self.calls = []

        def run(self, query, **parameters):
            self.calls.append((" ".join(query.split()), parameters))
            return FakeResult()

    session = FakeSession()
    replace_documents(session, ["document:one", "document:one", ""])

    assert len(session.calls) == 5
    assert session.calls[0][1]["document_ids"] == ["document:one"]
    assert "relationship.sourceChunkIds" in session.calls[1][0]
    assert "remaining_chunk_ids" in session.calls[1][0]
    assert "SET relationship.sourceChunkIds = remaining_chunk_ids" in session.calls[1][0]
    assert "WHERE size(remaining_chunk_ids) = 0 DELETE relationship" in session.calls[1][0]
    assert session.calls[1][1]["chunk_ids"] == ["chunk:old"]
    assert "DETACH DELETE chunk" in session.calls[2][0]
    assert "DETACH DELETE document" in session.calls[3][0]
    assert "DETACH DELETE entity" in session.calls[4][0]


def test_incremental_cli_upload_merges_relationship_provenance(monkeypatch, tmp_path):
    class FakeResult:
        def single(self):
            return {"chunk_ids": [], "entity_ids": []}

    class FakeSession:
        def __init__(self):
            self.queries = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run(self, query, **parameters):
            self.queries.append((" ".join(query.split()), parameters))
            return FakeResult()

    class FakeDriver:
        def __init__(self):
            self.session_instance = FakeSession()

        def session(self):
            return self.session_instance

        def close(self):
            return None

    driver = FakeDriver()
    graph_database = SimpleNamespace(driver=lambda *_args, **_kwargs: driver)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))

    output_dir = tmp_path / "graph"
    output_dir.mkdir()
    (output_dir / "knowledge_graph.json").write_text(json.dumps({
        "graph": {
            "nodes": [
                {"id": "entity:a", "name": "A", "type": "PERSON"},
                {"id": "entity:b", "name": "B", "type": "PERSON"},
            ],
            "links": [{
                "source": "entity:a",
                "target": "entity:b",
                "predicates": ["knows"],
                "relations": [{
                    "predicate": "knows",
                    "evidence_sentence": "A knows B.",
                    "source_chunk_id": "chunk:new",
                }],
            }],
        },
    }), encoding="utf-8")

    result = CliRunner().invoke(cli.main, ["neo4j-upload", "-o", str(output_dir)])

    assert result.exit_code == 0, result.output
    relationship_query = next(
        query for query, _ in driver.session_instance.queries
        if "MERGE (a)-[r:KNOWS]->(b)" in query
    )
    assert "reduce(ids = coalesce(r.sourceChunkIds, [])" in relationship_query
    assert "CASE WHEN chunk_id IN ids THEN ids ELSE ids + [chunk_id] END" in relationship_query
