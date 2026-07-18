"""Tests for entity and relation extraction."""

from types import SimpleNamespace

import builtins

import pytest

from kg_generator.config import Language
from kg_generator.extract.entities import Entity, SimpleExtractor, VietnameseExtractor
from kg_generator.extract.graphgen import GraphGenExtractor
from kg_generator.extract.relations import RelationExtractor


def test_simple_extractor_captures_capitalized():
    extractor = SimpleExtractor()
    text = "Alice and Bob visited New York City last summer."
    entities = extractor.extract(text)

    names = {e.name for e in entities}
    # SimpleExtractor with CAPITALIZED_PHRASE will find multi-word capitalized phrases
    assert "New York City" in names or any("Alice" in n for n in names) or any("Bob" in n for n in names)


def test_relation_preserves_stable_ids_evidence_and_source_chunk():
    alice = Entity(name="Alice", label="PERSON")
    acme = Entity(name="Acme Corp", label="ORG")

    relations = RelationExtractor().extract(
        "Alice works at Acme Corp. Another sentence.",
        [alice, acme],
        source_chunk_id="chunk:123",
    )

    assert len(relations) == 1
    subject, predicate, object_, evidence, source_chunk_id = relations[0]
    assert subject == alice.id
    assert object_ == acme.id
    assert predicate == "works_at"
    assert evidence == "Alice works at Acme Corp."
    assert source_chunk_id == "chunk:123"


def test_symmetric_relations_have_stable_direction_across_entity_order():
    first = Entity(name="dữ liệu", label="CONCEPT")
    second = Entity(name="trí tuệ nhân tạo", label="CONCEPT")
    extractor = RelationExtractor(language=Language.VIETNAMESE)
    text = "Dữ liệu và trí tuệ nhân tạo hỗ trợ nghiên cứu."

    forward = extractor.extract(text, [first, second])[0][:3]
    reversed_input = extractor.extract(text, [second, first])[0][:3]

    assert forward == reversed_input
    assert forward[1] == "related_to"


def test_vietnamese_extractor_reconstructs_bio_entities_and_noun_phrases():
    rows = [
        ("Đại học", "N", "B-NP", "B-ORG"),
        ("Quốc gia", "N", "I-NP", "I-ORG"),
        ("Hà Nội", "Np", "I-NP", "I-ORG"),
        ("nghiên cứu", "V", "B-VP", "O"),
        ("trí tuệ nhân tạo", "N", "B-NP", "O"),
        ("tại", "E", "B-PP", "O"),
        ("Việt Nam", "Np", "B-NP", "B-LOC"),
    ]
    extractor = VietnameseExtractor(ner_function=lambda _text: rows)

    entities = extractor.extract("Văn bản tiếng Việt")

    assert {(entity.name, entity.label) for entity in entities} == {
        ("Đại học Quốc gia Hà Nội", "ORG"),
        ("trí tuệ nhân tạo", "CONCEPT"),
        ("Việt Nam", "GPE"),
    }
    assert all("_" not in entity.name for entity in entities)


def test_vietnamese_extractor_recovers_from_malformed_i_tag():
    rows = [
        ("Nguyễn", "Np", "B-NP", "I-PER"),
        ("Trãi", "Np", "I-NP", "I-PER"),
    ]

    entities = VietnameseExtractor(ner_function=lambda _text: rows).extract("Nguyễn Trãi")

    assert [(entity.name, entity.label) for entity in entities] == [
        ("Nguyễn Trãi", "PERSON")
    ]


def test_vietnamese_extractor_fails_clearly_without_dependency(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "underthesea":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match="uv sync --extra vi"):
        VietnameseExtractor()


def test_real_underthesea_extracts_vietnamese_entities_when_installed():
    pytest.importorskip("underthesea")

    entities = VietnameseExtractor().extract(
        "Võ Nguyên Giáp sinh tại Quảng Bình và tham gia chiến dịch Điện Biên Phủ."
    )

    assert entities
    assert any(entity.name == "Quảng Bình" and entity.label == "GPE" for entity in entities)
    assert any(any(character in entity.name for character in "õăâđêôơư") for entity in entities)


class _FakeDeepSeekClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.responses)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_graphgen_jointly_extracts_entities_and_descriptive_relationships():
    text = "Alice founded Acme Corp in Hanoi."
    extraction = """("entity"<|>"Alice"<|>"person"<|>"A founder.")##
("entity"<|>"Acme Corp"<|>"organization"<|>"A company.")##
("entity"<|>"Hanoi"<|>"location"<|>"A city.")##
("relationship"<|>"Alice"<|>"Acme Corp"<|>"Alice founded Acme Corp.")##
("content_keywords"<|>"company founding")<|COMPLETE|>"""
    client = _FakeDeepSeekClient([extraction, "NO"])

    entities, relationships = GraphGenExtractor(client=client).extract(
        text, source_chunk_id="chunk:123"
    )

    assert [(entity.name, entity.label) for entity in entities] == [
        ("Alice", "PERSON"),
        ("Acme Corp", "ORGANIZATION"),
        ("Hanoi", "LOCATION"),
    ]
    assert len(relationships) == 1
    relation = relationships[0]
    assert relation[1] == "RELATION"
    assert relation[3] == ""
    assert relation[4] == "chunk:123"
    assert relation[5] == "Alice founded Acme Corp."
    assert len(relation) == 6
    assert "relationship_summary" in client.calls[0]["messages"][0]["content"]
    assert "response_format" not in client.calls[0]


def test_graphgen_iteratively_gleans_missed_records():
    initial = """("entity"<|>"Alice"<|>"person"<|>"A person.")<|COMPLETE|>"""
    glean = """("entity"<|>"Acme Corp"<|>"organization"<|>"A company.")##
("relationship"<|>"Alice"<|>"Acme Corp"<|>"Alice founded Acme Corp.")<|COMPLETE|>"""
    client = _FakeDeepSeekClient([initial, "YES", glean, "NO"])

    entities, relationships = GraphGenExtractor(client=client).extract("Source text.")

    assert {entity.name for entity in entities} == {"Alice", "Acme Corp"}
    assert len(relationships) == 1
    assert len(client.calls) == 4


def test_graphgen_retries_empty_response():
    extraction = """("entity"<|>"Alice"<|>"person"<|>"A person.")<|COMPLETE|>"""
    client = _FakeDeepSeekClient(["", extraction, "NO"])

    entities, relationships = GraphGenExtractor(client=client).extract("Some source text.")

    assert [entity.name for entity in entities] == ["Alice"]
    assert relationships == []
    assert len(client.calls) == 3


def test_graphgen_aggregates_repeated_descriptions_with_figure_9():
    client = _FakeDeepSeekClient([
        "Alice is a scientist and Nobel Prize winner.",
        "Alice founded Acme Corp and later led it.",
    ])
    extractor = GraphGenExtractor(client=client, max_gleanings=0)
    resolved = [{"id": "entity:alice", "name": "Alice", "description": "first"}]
    originals = [
        {"id": "entity:alice", "description": "Alice is a scientist."},
        {"id": "entity:alice", "description": "Alice won a Nobel Prize."},
    ]
    triples = [
        ("entity:alice", "RELATION", "entity:acme", "", "chunk:1", "Alice founded Acme."),
        ("entity:alice", "RELATION", "entity:acme", "", "chunk:2", "Alice led Acme."),
    ]

    entities, relations = extractor.aggregate_descriptions(
        resolved, originals, {}, triples
    )

    assert entities[0]["description"] == "Alice is a scientist and Nobel Prize winner."
    assert {relation[5] for relation in relations} == {
        "Alice founded Acme Corp and later led it."
    }
    assert "Description List" in client.calls[0]["messages"][0]["content"]


def test_graphgen_uses_vietnamese_prompt_pack_and_preserves_diacritics():
    extraction = """("entity"<|>"Võ Nguyên Giáp"<|>"person"<|>"Một vị tướng Việt Nam.")##
("entity"<|>"Điện Biên Phủ"<|>"event"<|>"Một chiến dịch lịch sử.")##
("relationship"<|>"Võ Nguyên Giáp"<|>"Điện Biên Phủ"<|>"Ông tham gia chỉ huy chiến dịch.")<|COMPLETE|>"""
    client = _FakeDeepSeekClient([extraction, "NO"])
    extractor = GraphGenExtractor(
        language=Language.VIETNAMESE,
        client=client,
    )

    entities, relationships = extractor.extract("Võ Nguyên Giáp chỉ huy Điện Biên Phủ.")

    assert [entity.name for entity in entities] == ["Võ Nguyên Giáp", "Điện Biên Phủ"]
    assert relationships[0][5] == "Ông tham gia chỉ huy chiến dịch."
    prompt = client.calls[0]["messages"][0]["content"]
    assert "Bạn là chuyên gia NLP về tiếng Việt" in prompt
    assert "Đại học Quốc gia Hà Nội" in prompt
    assert extractor.prompt_version == "graphgen-figure8-figure9-v2"


def test_graphgen_uses_vietnamese_summary_prompt():
    client = _FakeDeepSeekClient(["Võ Nguyên Giáp là một nhân vật lịch sử Việt Nam."])
    extractor = GraphGenExtractor(
        language=Language.VIETNAMESE,
        client=client,
        max_gleanings=0,
    )

    extractor.aggregate_descriptions(
        [{"id": "entity:giap", "name": "Võ Nguyên Giáp", "description": ""}],
        [
            {"id": "entity:giap", "description": "Một vị tướng."},
            {"id": "entity:giap", "description": "Một nhân vật lịch sử."},
        ],
        {},
        [],
    )

    assert "Danh sách mô tả" in client.calls[0]["messages"][0]["content"]
