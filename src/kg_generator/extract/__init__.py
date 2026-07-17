"""Stage 2: Entity and relation extraction."""

from kg_generator.extract.entities import Entity, EntityExtractor, EnglishExtractor, VietnameseExtractor
from kg_generator.extract.relations import RelationExtractor

__all__ = [
    "Entity",
    "EntityExtractor",
    "EnglishExtractor",
    "VietnameseExtractor",
    "RelationExtractor",
]
