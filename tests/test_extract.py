"""Tests for entity and relation extraction."""

from types import SimpleNamespace

from kg_generator.extract.entities import Entity, SimpleExtractor
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

    assert relations
    subject, predicate, object_, evidence, source_chunk_id = relations[0]
    assert subject in {alice.id, acme.id}
    assert object_ in {alice.id, acme.id}
    assert predicate
    assert evidence == "Alice works at Acme Corp."
    assert source_chunk_id == "chunk:123"


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
