"""Stage 3: Entity resolution and disambiguation."""

from kg_generator.resolve.resolver import EntityResolver
from kg_generator.resolve.neo4j_resolver import Neo4jEntityResolver

__all__ = ["EntityResolver", "Neo4jEntityResolver"]
