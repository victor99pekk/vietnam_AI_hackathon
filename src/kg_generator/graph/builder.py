"""Graph construction from resolved entities and triples."""

import logging
from typing import Any

import networkx as nx

from kg_generator.config import GraphBackend, Ontology

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds a knowledge graph from entities and relation triples."""

    def __init__(
        self,
        ontology: Ontology | None = None,
        backend: GraphBackend = GraphBackend.NETWORKX,
    ) -> None:
        self.ontology = ontology
        self.backend = backend

    def build(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> nx.DiGraph:
        """Build a directed graph from entities and (subject, predicate, object, source_text) triples."""
        graph = nx.DiGraph()

        # Add entity nodes with GraphRAG properties
        for entity in entities:
            graph.add_node(
                entity["name"],
                id=entity.get("id", ""),
                name=entity.get("name", ""),
                type=entity.get("type", "ENTITY"),
                aliases=entity.get("aliases", []),
                description=entity.get("description", ""),
                importanceScore=entity.get("importanceScore", 0.0),
                confidenceScore=entity.get("confidenceScore", 1.0),
                source=entity.get("source", []),
                embedding=entity.get("embedding"),
                updatedAt=entity.get("updatedAt", ""),
                # Chunk-specific
                text=entity.get("text", ""),
                tokenCount=entity.get("tokenCount", 0),
                index=entity.get("index", 0),
                chunk_count=entity.get("chunk_count", 0),
            )

        # Add relation edges
        for triple in triples:
            subj, pred, obj = triple[0], triple[1], triple[2]
            source_text = triple[3] if len(triple) > 3 else ""

            # Ensure both endpoints exist as nodes
            if subj not in graph:
                graph.add_node(subj, label="ENTITY", type="ENTITY", name=subj)
            if obj not in graph:
                graph.add_node(obj, label="ENTITY", type="ENTITY", name=obj)

            # Add or update the edge
            if graph.has_edge(subj, obj):
                existing = graph.edges[subj, obj].get("predicates", [])
                if pred not in existing:
                    existing.append(pred)
                graph.edges[subj, obj]["predicates"] = existing
                graph.edges[subj, obj]["weight"] = len(existing)
                existing_sources = graph.edges[subj, obj].get("source_texts", [])
                if source_text and source_text not in existing_sources:
                    existing_sources.append(source_text)
                    graph.edges[subj, obj]["source_texts"] = existing_sources
            else:
                graph.add_edge(
                    subj, obj,
                    predicates=[pred],
                    weight=1,
                    source_texts=[source_text] if source_text else [],
                )

        # Compute PageRank-based importance scores
        self._compute_importance(graph)

        logger.info(
            f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

        # Validate against ontology if provided
        if self.ontology:
            self._validate(graph)

        return graph

    @staticmethod
    def _compute_importance(graph: nx.DiGraph) -> None:
        """Compute PageRank importance and store as importanceScore on each node."""
        try:
            pr = nx.pagerank(graph, max_iter=100, tol=1e-4)
            for node, score in pr.items():
                graph.nodes[node]["importanceScore"] = round(score, 6)
        except Exception:
            logger.debug("PageRank computation failed — skipping importance scores")

    def _validate(self, graph: nx.DiGraph) -> None:
        """Check graph consistency against the ontology schema."""
        if not self.ontology:
            return

        ontology_labels = set(self.ontology.entity_types.keys())
        ontology_relations = set(self.ontology.relationship_types.keys())

        node_label_mismatches = 0
        edge_relation_mismatches = 0

        for _, data in graph.nodes(data=True):
            label = data.get("label", "")
            if label and label not in ontology_labels:
                node_label_mismatches += 1

        for _, _, data in graph.edges(data=True):
            predicates = data.get("predicates", [])
            for p in predicates:
                if p not in ontology_relations:
                    edge_relation_mismatches += 1

        if node_label_mismatches:
            logger.warning(f"Ontology validation: {node_label_mismatches} node labels not in schema")
        if edge_relation_mismatches:
            logger.warning(f"Ontology validation: {edge_relation_mismatches} relation types not in schema")

    def stats(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Return summary statistics for the graph."""
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "num_connected_components": nx.number_weakly_connected_components(graph),
            "avg_degree": sum(dict(graph.degree()).values()) / max(graph.number_of_nodes(), 1),
            "label_distribution": self._label_distribution(graph),
            "relation_distribution": self._relation_distribution(graph),
        }

    @staticmethod
    def _label_distribution(graph: nx.DiGraph) -> dict[str, int]:
        dist: dict[str, int] = {}
        for _, data in graph.nodes(data=True):
            label = data.get("label", "UNKNOWN")
            dist[label] = dist.get(label, 0) + 1
        return dist

    @staticmethod
    def _relation_distribution(graph: nx.DiGraph) -> dict[str, int]:
        dist: dict[str, int] = {}
        for _, _, data in graph.edges(data=True):
            for pred in data.get("predicates", []):
                dist[pred] = dist.get(pred, 0) + 1
        return dist
