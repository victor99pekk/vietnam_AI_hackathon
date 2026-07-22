"""Tests for the isolated GraphGen-style evaluation method."""

import json

from kg_generator.evaluate.graphgen.qa_generator import GraphGenQAGenerator
from kg_generator.evaluate.graphgen.subgraphs import (
    GraphGenSubgraphSampler,
    KnowledgeEdge,
    load_graphgen_kg,
)


def _nodes():
    return {
        name: {
            "id": name,
            "name": name,
            "type": "ENTITY",
            "description": f"Description of {name}",
        }
        for name in ("a", "b", "c", "d", "e")
    }


def _edge(edge_id, source, target, loss):
    return KnowledgeEdge(
        id=edge_id,
        source=source,
        target=target,
        description=f"{source} is connected to {target}",
        source_chunk_ids=("chunk:1",),
        loss=loss,
    )


def test_loader_reads_descriptive_triples_and_excludes_structure(tmp_path):
    kg_path = tmp_path / "knowledge_graph.json"
    kg_path.write_text(
        json.dumps(
            {
                "graph": {
                    "nodes": [
                        {"id": "a", "name": "A", "type": "PERSON"},
                        {"id": "b", "name": "B", "type": "PLACE"},
                        {"id": "chunk:1", "type": "Chunk"},
                    ]
                },
                "triples": [
                    {
                        "subject": "a",
                        "predicate": "RELATION",
                        "object": "b",
                        "description": "A worked at B.",
                        "source_chunk_id": "chunk:1",
                    },
                    {
                        "subject": "chunk:1",
                        "predicate": "MENTIONS",
                        "object": "a",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    nodes, edges = load_graphgen_kg(kg_path)

    assert set(nodes) == {"a", "b"}
    assert len(edges) == 1
    assert edges[0].description == "A worked at B."
    assert edges[0].source_chunk_ids == ("chunk:1",)


def test_khop_sampler_is_deterministic_and_audits_selection():
    edges = [
        _edge("edge:ab", "a", "b", 0.9),
        _edge("edge:bc", "b", "c", 0.8),
        _edge("edge:cd", "c", "d", 0.7),
        _edge("edge:ae", "a", "e", 0.1),
    ]
    sampler = GraphGenSubgraphSampler(
        max_depth=1,
        max_premise_tokens=200,
        max_extra_edges=2,
        edge_sampling="max_loss",
    )

    first = sampler.sample(_nodes(), edges, max_subgraphs=1)
    second = sampler.sample(_nodes(), edges, max_subgraphs=1)

    assert first.subgraphs == second.subgraphs
    assert first.audit == second.audit
    assert first.subgraphs[0]["seed_edge_id"] == "edge:ab"
    assert {edge["id"] for edge in first.subgraphs[0]["edges"]} == {
        "edge:ab",
        "edge:bc",
        "edge:ae",
    }
    assert first.subgraphs[0]["selection_basis"] == "comprehension_loss"
    assert any(event["reason"] == "depth_limit" for event in first.audit)


def test_sampler_records_missing_loss_fallback():
    edges = [
        _edge("edge:ab", "a", "b", None),
        _edge("edge:bc", "b", "c", None),
    ]
    sampler = GraphGenSubgraphSampler(max_depth=1, max_premise_tokens=200)

    result = sampler.sample(_nodes(), edges, max_subgraphs=1)

    assert result.subgraphs[0]["selection_basis"] == (
        "stable_id_fallback_no_comprehension_loss"
    )
    assert any(event["reason"] == "stable_id_fallback" for event in result.audit)


def test_qa_generator_requires_and_logs_two_used_edges(tmp_path):
    subgraph = {
        "id": "subgraph:1",
        "nodes": [
            {"id": "a", "name": "Alan", "type": "PERSON", "description": ""},
            {"id": "b", "name": "Park", "type": "PLACE", "description": ""},
            {"id": "c", "name": "School", "type": "ORG", "description": ""},
        ],
        "edges": [
            {
                "id": "edge:1",
                "source": "a",
                "target": "b",
                "description": "Alan worked at Park.",
                "source_chunk_ids": ["chunk:1"],
            },
            {
                "id": "edge:2",
                "source": "b",
                "target": "c",
                "description": "School was located at Park.",
                "source_chunk_ids": ["chunk:1"],
            },
        ],
        "source_chunk_ids": ["chunk:1"],
    }
    response = json.dumps(
        {
            "question": "Which school was at the place where Alan worked?",
            "answer": "School was at Park, where Alan worked.",
            "used_edge_ids": ["edge:1", "edge:2"],
        }
    )
    generator = GraphGenQAGenerator(llm_call=lambda _prompt: response)

    qa_path, audit_path = generator.generate([subgraph], tmp_path)

    pair = json.loads(qa_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert pair["scenario"] == "multi_hop"
    assert pair["used_edge_ids"] == ["edge:1", "edge:2"]
    assert pair["source_chunk_ids"] == ["chunk:1"]
    assert audit["decision"] == "accepted"
