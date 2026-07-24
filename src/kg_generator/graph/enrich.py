"""Graph enrichment — linking to external knowledge bases."""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


class GraphEnricher:
    """Enriches a knowledge graph with data from external sources (Wikidata, DBpedia, etc.)."""

    def __init__(self) -> None:
        pass

    def enrich(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Enrich the graph with external data.

        Currently not implemented. Planned features:
        - Wikidata entity linking for disambiguated nodes
        - DBpedia SPARQL queries for additional attributes
        - Cross-graph entity alignment
        """
        raise NotImplementedError(
            "External KB enrichment is not yet implemented. "
            "See src/kg_generator/graph/enrich.py for planned features."
        )

    def link_wikidata(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Link entities to Wikidata Q-IDs and import relevant attributes."""
        raise NotImplementedError(
            "Wikidata linking not yet implemented. "
            "Planned: use qwikidata or direct Wikidata API to resolve entity names to Q-IDs."
        )

    def link_dbpedia(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Query DBpedia for additional facts about entities."""
        raise NotImplementedError(
            "DBpedia linking not yet implemented. "
            "Planned: SPARQL queries against dbpedia.org/sparql for additional attributes."
        )
