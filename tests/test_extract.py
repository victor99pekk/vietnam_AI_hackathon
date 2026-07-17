"""Tests for entity and relation extraction."""

from kg_generator.extract.entities import Entity, SimpleExtractor


def test_simple_extractor_captures_capitalized():
    extractor = SimpleExtractor()
    text = "Alice and Bob visited New York City last summer."
    entities = extractor.extract(text)

    names = {e.name for e in entities}
    # SimpleExtractor with CAPITALIZED_PHRASE will find multi-word capitalized phrases
    assert "New York City" in names or any("Alice" in n for n in names) or any("Bob" in n for n in names)
