"""Stage 4: Graph construction and enrichment."""

from kg_generator.config import GraphBackend
from kg_generator.graph.builder import GraphBuilder
from kg_generator.graph.enrich import GraphEnricher
from kg_generator.graph.neo4j_builder import Neo4jGraphBuilder

__all__ = ["GraphBuilder", "GraphEnricher", "Neo4jGraphBuilder", "get_graph_builder"]


def get_graph_builder(backend: GraphBackend, **kwargs):
    """Return the appropriate graph builder for a given backend."""
    if backend == GraphBackend.NEO4J:
        return Neo4jGraphBuilder(**kwargs)
    return GraphBuilder(ontology=kwargs.get("ontology"), backend=backend)
