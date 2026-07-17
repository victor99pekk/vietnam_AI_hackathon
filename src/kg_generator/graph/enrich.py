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

        Currently a stub — will add:
        - Wikidata entity linking for disambiguated nodes
        - DBpedia SPARQL queries for additional attributes
        - Cross-graph entity alignment
        """
        logger.info("Graph enrichment: no external sources configured — returning graph as-is")
        return graph

    def link_wikidata(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Link entities to Wikidata Q-IDs and import relevant attributes. (Stub)"""
        # TODO: Implement Wikidata API lookup
        # from qwikidata.linked_data_interface import get_entity_dict_from_api
        logger.warning("Wikidata linking not yet implemented")
        return graph

    def link_dbpedia(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Query DBpedia for additional facts about entities. (Stub)"""
        # TODO: Implement SPARQL queries
        logger.warning("DBpedia linking not yet implemented")
        return graph
