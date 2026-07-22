"""Quality evaluation metrics for knowledge graphs."""

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """
    Evaluates KG quality against metrics from the Problem Description:
    completeness, consistency, duplication level, missing information,
    format errors, labeling quality, and reusability.
    """

    def evaluate(
        self,
        path: Path,
    ) -> dict[str, float]:
        """Evaluate a serialized KG from a file path."""
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            entities = data.get("entities", [])
            triples = data.get("triples", [])
            graph = nx.node_link_graph(data.get("graph", {}))
        else:
            logger.warning(f"Unsupported format: {path.suffix}")
            return {}

        return self.evaluate_graph(graph, entities, triples)

    def evaluate_graph(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Compute all quality metrics for a graph."""
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_triples": len(triples),
            **self.completeness(entities),
            **self.consistency(graph, triples),
            **self.duplication_level(entities, triples),
            **self.missing_information(entities),
            **self.format_errors(triples),
            **self.labeling_quality(entities, graph),
            **self.reusability_score(graph, entities, triples),
            "overall_score": self._overall_score(graph, entities, triples),
        }

    # ── Individual Metrics ──

    def completeness(self, entities: list[dict[str, Any]]) -> dict[str, float]:
        """Fraction of entities that have all key fields populated."""
        if not entities:
            return {"completeness": 0.0, "completeness_breakdown": {}}

        expected_fields = {"name", "type", "aliases"}
        scores = []
        breakdown: dict[str, float] = {}

        for field in expected_fields:
            filled = sum(1 for e in entities if e.get(field))
            score = filled / len(entities)
            breakdown[f"has_{field}"] = score
            scores.append(score)

        return {
            "completeness": sum(scores) / len(scores),
            "completeness_breakdown": breakdown,
        }

    def consistency(
        self,
        graph: nx.DiGraph,
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, float]:
        """Schema conformance and structural consistency."""
        if not triples:
            return {"consistency": 1.0}

        node_names = set(graph.nodes())
        orphan_count = sum(
            1 for t in triples
            if t[0] not in node_names or t[2] not in node_names
        )
        endpoint_score = 1.0 - (orphan_count / len(triples))

        label_conflicts = 0
        for node in graph.nodes():
            data = graph.nodes[node]
            if data.get("type") in ("UNKNOWN", "", None):
                label_conflicts += 1
        label_score = 1.0 - (label_conflicts / max(graph.number_of_nodes(), 1))

        score = (endpoint_score + label_score) / 2
        return {"consistency": score}

    def duplication_level(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, float]:
        """Detect duplicate entities and triples."""
        if not entities or not triples:
            return {"duplication_level": 0.0}

        # Entity name duplication
        names = [e["name"].lower() for e in entities]
        entity_dup_ratio = 1.0 - (len(set(names)) / len(names))

        # Triple duplication (ignore source_text for dedup comparison)
        triple_keys = {(t[0], t[1], t[2]) for t in triples}
        triple_dup_ratio = 1.0 - (len(triple_keys) / len(triples))

        score = (entity_dup_ratio + triple_dup_ratio) / 2
        return {
            "duplication_level": score,
            "entity_duplication_rate": entity_dup_ratio,
            "triple_duplication_rate": triple_dup_ratio,
        }

    def missing_information(self, entities: list[dict[str, Any]]) -> dict[str, float]:
        """Fraction of entities with empty/missing type or aliases."""
        if not entities:
            return {"missing_information": 1.0}

        missing_count = sum(
            1 for e in entities
            if not e.get("type") or not e.get("aliases") or e.get("type") in ("UNKNOWN", "")
        )
        return {"missing_information": missing_count / len(entities)}

    def format_errors(self, triples: list[tuple[str, str, str, str]]) -> dict[str, float]:
        """Detect malformed triples (empty strings, wrong types)."""
        if not triples:
            return {"format_errors": 0.0}

        errors = 0
        for t in triples:
            if not t[0] or not t[1] or not t[2]:
                errors += 1
            elif not isinstance(t[0], str) or not isinstance(t[1], str) or not isinstance(t[2], str):
                errors += 1

        return {"format_errors": errors / len(triples)}

    def labeling_quality(
        self,
        entities: list[dict[str, Any]],
        graph: nx.DiGraph,
    ) -> dict[str, float]:
        """Heuristic labeling quality — fraction of entities with meaningful types."""
        if not entities:
            return {"labeling_quality": 0.0}

        generic = {"ENTITY", "UNKNOWN", "NAMED_ENTITY", "CONCEPT"}
        meaningful = sum(
            1 for e in entities
            if e.get("type") and e["type"] not in generic
        )
        return {"labeling_quality": meaningful / len(entities)}

    def reusability_score(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, float]:
        """Score indicating how reusable the KG is for downstream tasks."""
        score = 0.0

        if graph.number_of_nodes() > 0:
            score += 0.2
        if graph.number_of_edges() > 0:
            score += 0.2

        if graph.number_of_edges() > 0:
            connected_ratio = (
                len(max(nx.weakly_connected_components(graph), key=len))
                / max(graph.number_of_nodes(), 1)
            )
            score += 0.2 * connected_ratio

        # Has meaningful types (not generic fallbacks)
        generic = {"ENTITY", "UNKNOWN", "Chunk", "Document"}
        if entities:
            meaningful = sum(1 for e in entities if e.get("type") not in generic)
            score += 0.2 * (meaningful / len(entities))

        # Has descriptions
        if entities:
            has_desc = sum(1 for e in entities if e.get("description"))
            score += 0.2 * (has_desc / len(entities))

        return {"reusability": score}

    def _overall_score(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> float:
        # Compute each metric individually (not via evaluate_graph to avoid recursion)
        comp = self.completeness(entities).get("completeness", 0.0)
        cons = self.consistency(graph, triples).get("consistency", 0.0)
        dup = self.duplication_level(entities, triples).get("duplication_level", 0.0)
        missing = self.missing_information(entities).get("missing_information", 0.0)
        fmt_err = self.format_errors(triples).get("format_errors", 0.0)
        label_q = self.labeling_quality(entities, graph).get("labeling_quality", 0.0)
        reuse = self.reusability_score(graph, entities, triples).get("reusability", 0.0)

        scores = [comp, cons, label_q, reuse, 1.0 - dup, 1.0 - fmt_err, 1.0 - missing]
        return sum(scores) / len(scores)
