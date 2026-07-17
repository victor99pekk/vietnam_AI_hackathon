"""Graph construction from resolved entities and triples."""

import logging
from typing import Any

import networkx as nx

from kg_generator.config import GraphBackend, Ontology
from kg_generator.identity import entity_id

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
        triples: list[tuple[str, ...]],
    ) -> nx.DiGraph:
        """Build a graph from ID-based triples with evidence and chunk provenance."""
        graph = nx.DiGraph()

        # IDs are graph keys. Names are display properties only.
        for entity in entities:
            node_type = entity.get("type", entity.get("label", "ENTITY"))
            node_id = entity.get("id") or entity_id(node_type, entity.get("name", ""))
            graph.add_node(
                node_id,
                id=node_id,
                name=entity.get("name", ""),
                type=node_type,
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
            evidence_sentence = triple[3] if len(triple) > 3 else ""
            source_chunk_id = triple[4] if len(triple) > 4 else ""
            relation_record = {
                "predicate": pred,
                "evidence_sentence": evidence_sentence,
                "source_chunk_id": source_chunk_id,
            }
            if len(triple) > 5:
                relation_record["description"] = triple[5]
            if len(triple) > 6:
                relation_record["confidenceScore"] = triple[6]

            # Ensure both endpoints exist as nodes
            if subj not in graph:
                graph.add_node(subj, id=subj, type="ENTITY", name=subj)
            if obj not in graph:
                graph.add_node(obj, id=obj, type="ENTITY", name=obj)

            # Add or update the edge
            if graph.has_edge(subj, obj):
                existing = graph.edges[subj, obj].get("predicates", [])
                if pred not in existing:
                    existing.append(pred)
                graph.edges[subj, obj]["predicates"] = existing
                graph.edges[subj, obj]["weight"] = len(existing)
                existing_sources = graph.edges[subj, obj].get("source_texts", [])
                if evidence_sentence and evidence_sentence not in existing_sources:
                    existing_sources.append(evidence_sentence)
                    graph.edges[subj, obj]["source_texts"] = existing_sources
                source_chunks = graph.edges[subj, obj].get("source_chunk_ids", [])
                if source_chunk_id and source_chunk_id not in source_chunks:
                    source_chunks.append(source_chunk_id)
                    graph.edges[subj, obj]["source_chunk_ids"] = source_chunks
                relation_records = graph.edges[subj, obj].get("relations", [])
                if relation_record not in relation_records:
                    relation_records.append(relation_record)
                    graph.edges[subj, obj]["relations"] = relation_records
                if relation_record.get("description"):
                    graph.edges[subj, obj]["description"] = relation_record["description"]
            else:
                graph.add_edge(
                    subj, obj,
                    predicates=[pred],
                    weight=1,
                    source_texts=[evidence_sentence] if evidence_sentence else [],
                    source_chunk_ids=[source_chunk_id] if source_chunk_id else [],
                    relations=[relation_record],
                    description=relation_record.get("description", ""),
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
            label = data.get("type", data.get("label", ""))
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
            label = data.get("type", data.get("label", "UNKNOWN"))
            dist[label] = dist.get(label, 0) + 1
        return dist

    @staticmethod
    def _relation_distribution(graph: nx.DiGraph) -> dict[str, int]:
        dist: dict[str, int] = {}
        for _, _, data in graph.edges(data=True):
            for pred in data.get("predicates", []):
                dist[pred] = dist.get(pred, 0) + 1
        return dist
