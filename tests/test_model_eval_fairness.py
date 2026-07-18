import json
from pathlib import Path

import networkx as nx

from evaluation.model_eval.dataset_gen import (
    QADatasetGenerator,
    balance_jsonl_token_volume,
    estimate_qa_tokens,
    load_kg,
    load_raw_documents_from_kg,
)


def _write_kg(path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_node("a", id="a", name="An", type="Person")
    graph.add_node("b", id="b", name="Bình", type="Location")
    graph.add_node(
        "chunk:1",
        id="chunk:1",
        type="Chunk",
        text="An sinh ra ở Bình.",
        source="source.jsonl",
    )
    graph.add_edge("a", "b", predicates=["born_in"])
    payload = {
        "entities": [
            {"id": "a", "name": "An", "type": "Person"},
            {"id": "b", "name": "Bình", "type": "Location"},
        ],
        "graph": nx.node_link_data(graph, edges="edges"),
        "triples": [
            {
                "subject": "a",
                "predicate": "born_in",
                "object": "b",
                "evidence_sentence": "An sinh ra ở Bình.",
                "source_chunk_id": "chunk:1",
            },
            {
                "subject": "chunk:1",
                "predicate": "PART_OF",
                "object": "document:1",
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_current_dict_triples_and_embedded_raw_chunks_are_loaded(tmp_path: Path):
    kg_path = tmp_path / "knowledge_graph.json"
    _write_kg(kg_path)

    graph, entities, triples = load_kg(kg_path)
    raw_documents = load_raw_documents_from_kg(kg_path)

    assert graph.number_of_edges() == 1
    assert len(entities) == 2
    assert triples == [("a", "born_in", "b", "An sinh ra ở Bình.", "chunk:1")]
    assert raw_documents == [{"content": "An sinh ra ở Bình.", "source": "chunk:1"}]


def test_grouped_split_keeps_fact_paraphrases_out_of_both_splits(tmp_path: Path):
    graph = nx.DiGraph()
    entities = []
    triples = []
    for index in range(10):
        subject = f"person:{index}"
        obj = f"place:{index}"
        graph.add_node(subject, name=f"Người {index}", type="Person")
        graph.add_node(obj, name=f"Nơi {index}", type="Location")
        graph.add_edge(subject, obj, predicates=["born_in"])
        entities.extend([
            {"id": subject, "name": f"Người {index}", "type": "Person"},
            {"id": obj, "name": f"Nơi {index}", "type": "Location"},
        ])
        triples.append((subject, "born_in", obj, f"Bằng chứng {index}", f"chunk:{index}"))

    generator = QADatasetGenerator(language="vi", seed=7, max_hops=1, test_split=0.2)
    train_path, test_path = generator.generate_from_kg(graph, entities, triples, tmp_path)
    raw_train_path, raw_test_path = generator.generate_from_raw_text(
        [
            {"content": f"Người {index} sinh ra ở Nơi {index}.", "source": f"chunk:{index}"}
            for index in range(10)
        ],
        tmp_path,
    )

    train = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
    raw_train = [json.loads(line) for line in raw_train_path.read_text(encoding="utf-8").splitlines()]
    raw_test = [json.loads(line) for line in raw_test_path.read_text(encoding="utf-8").splitlines()]
    train_evidence = {item["evidence"] for item in train if item.get("evidence")}
    test_evidence = {item["evidence"] for item in test if item.get("evidence")}
    assert train_evidence.isdisjoint(test_evidence)
    kg_train_sources = {source for item in train for source in item.get("source_chunk_ids", [])}
    kg_test_sources = {source for item in test for source in item.get("source_chunk_ids", [])}
    assert kg_train_sources.isdisjoint({item["source_id"] for item in raw_test})
    assert kg_test_sources.isdisjoint({item["source_id"] for item in raw_train})


def test_raw_generator_rejects_pronoun_as_factual_subject():
    generator = QADatasetGenerator(language="en", seed=42)
    pairs = generator._extract_qa_from_text(
        "He studied at King's College. Alan Turing studied at Cambridge University.",
        "source",
    )
    questions = {pair["question"] for pair in pairs}
    assert "Where did He study?" not in questions
    assert "Where did Alan Turing study?" in questions


def test_training_files_are_balanced_by_estimated_tokens(tmp_path: Path):
    kg_path = tmp_path / "kg.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    kg_items = [
        {"question": f"Question {index}", "answer": "short answer"}
        for index in range(30)
    ]
    raw_items = [
        {"question": f"Raw {index}", "answer": "a somewhat longer source answer"}
        for index in range(8)
    ]
    for path, items in ((kg_path, kg_items), (raw_path, raw_items)):
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in items),
            encoding="utf-8",
        )

    report = balance_jsonl_token_volume(kg_path, raw_path)
    kg_tokens = report["kg"]["estimated_tokens"]
    raw_tokens = report["raw"]["estimated_tokens"]
    largest_pair = max(estimate_qa_tokens(item) for item in kg_items + raw_items)
    assert abs(kg_tokens - raw_tokens) <= largest_pair
    assert report["kg"]["examples"] < len(kg_items)
