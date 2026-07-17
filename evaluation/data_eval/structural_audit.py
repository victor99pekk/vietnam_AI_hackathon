"""
Method 1, Step 1 — Intrinsic Structural Audit

Deep-dive graph health check focused on "graph rot" that would poison
SFT training data. Goes beyond basic metrics (completeness, consistency)
to catch issues specific to LLM training data quality.

Checks:
  - Orphan rate (nodes with zero connections)
  - Graph density (too sparse or too dense?)
  - Schema/ontology compliance (do edges follow defined rules?)
  - Entity duplication (unmerged near-duplicate nodes via embeddings)
  - Multi-hop connectivity (can the graph support chain reasoning?)
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

logger = logging.getLogger(__name__)


class StructuralAuditor:
    """Audits a knowledge graph for structural issues that would degrade SFT quality."""

    def __init__(
        self,
        ontology_path: Path | None = None,
        entity_dedup_threshold: float = 0.85,
    ) -> None:
        self.ontology_path = ontology_path
        self.entity_dedup_threshold = entity_dedup_threshold
        self._ontology: dict[str, Any] | None = None

    @property
    def ontology(self) -> dict[str, Any]:
        if self._ontology is None and self.ontology_path:
            with open(self.ontology_path) as f:
                self._ontology = yaml.safe_load(f)
        return self._ontology or {}

    def audit(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Run full structural audit and return a report dict."""
        report: dict[str, Any] = {
            "graph_stats": self._basic_stats(graph, triples),
            "orphan_analysis": self._orphan_analysis(graph, entities),
            "density_analysis": self._density_analysis(graph),
            "schema_compliance": self._schema_compliance(graph, triples),
            "entity_duplication": self._entity_duplication(entities),
            "multi_hop_connectivity": self._multi_hop_connectivity(graph),
            "overall_health_score": 0.0,  # computed below
        }

        # Compute overall health score (0-100)
        scores = [
            report["orphan_analysis"]["health_score"],
            report["density_analysis"]["health_score"],
            report["schema_compliance"]["health_score"],
            report["entity_duplication"]["health_score"],
            report["multi_hop_connectivity"]["health_score"],
        ]
        report["overall_health_score"] = round(sum(scores) / len(scores), 1)

        # Add interpretation
        report["verdict"] = self._interpret(report["overall_health_score"])
        return report

    # ── Individual Audit Functions ────────────────────────────

    @staticmethod
    def _basic_stats(
        graph: nx.DiGraph, triples: list[tuple[str, str, str, str]]
    ) -> dict[str, Any]:
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_triples": len(triples),
            "is_directed": graph.is_directed(),
            "is_connected": nx.is_weakly_connected(graph) if graph.number_of_nodes() > 0 else False,
        }

    def _orphan_analysis(
        self, graph: nx.DiGraph, entities: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Identify nodes with zero connections (orphans)."""
        n_nodes = graph.number_of_nodes()
        if n_nodes == 0:
            return {"orphan_count": 0, "orphan_rate": 0.0, "health_score": 100}

        orphans = list(nx.isolates(graph))
        orphan_count = len(orphans)
        orphan_rate = orphan_count / n_nodes

        # Filter to only entity nodes (not Chunk/Document) for actionable orphans
        entity_orphans = [
            n for n in orphans
            if graph.nodes[n].get("type") not in ("Chunk", "Document")
        ]
        chunk_orphans = len(orphans) - len(entity_orphans)

        # Health score: 100 = no orphans, 0 = all orphans (harsh above 30%)
        health = max(0, 100 - (orphan_rate * 300))

        return {
            "orphan_count": orphan_count,
            "orphan_rate": round(orphan_rate, 4),
            "entity_orphans": len(entity_orphans),
            "chunk_orphans": chunk_orphans,
            "health_score": round(health, 1),
            "flag": "red" if orphan_rate > 0.3 else ("yellow" if orphan_rate > 0.1 else "green"),
            "recommendation": (
                "High orphan rate — consider merging orphan entities or enriching connections."
                if orphan_rate > 0.3
                else ""
            ),
        }

    @staticmethod
    def _density_analysis(graph: nx.DiGraph) -> dict[str, Any]:
        """Graph density: too sparse = isolated facts; too dense = over-connected noise."""
        n = graph.number_of_nodes()
        if n < 2:
            return {"density": 0.0, "health_score": 100}

        density = nx.density(graph)

        # For a KG: ideal density is typically 0.01-0.05 (sparse but connected)
        # Too sparse (<0.005) = dead facts; too dense (>0.2) = possible noise
        if density < 0.005:
            health = max(0, density * 2000)  # scale up
        elif density <= 0.2:
            health = 100  # sweet spot
        else:
            health = max(0, 100 - (density - 0.2) * 500)

        return {
            "density": round(density, 6),
            "health_score": round(health, 1),
            "flag": "red" if density < 0.005 else ("yellow" if density > 0.2 else "green"),
            "recommendation": (
                "Graph is very sparse — relations may not capture enough context for multi-hop reasoning."
                if density < 0.005
                else (
                    "Graph is very dense — possible over-connected noise; verify relation quality."
                    if density > 0.2
                    else ""
                )
            ),
        }

    def _schema_compliance(
        self,
        graph: nx.DiGraph,
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Check if edges follow ontology schema rules (e.g., works_at: PERSON→ORG)."""
        if not self.ontology or not triples:
            return {
                "compliance_rate": 1.0,
                "violations": [],
                "health_score": 100,
                "flag": "green",
            }

        rel_rules = self.ontology.get("relationship_types", {})
        if not rel_rules:
            return {
                "compliance_rate": 1.0,
                "violations": [],
                "health_score": 100,
                "flag": "green",
            }

        violations: list[dict[str, str]] = []
        for subj, pred, obj, _ in triples:
            rule = rel_rules.get(pred, {})
            if not rule:
                continue  # No rule defined = no violation

            expected_domain = rule.get("domain", "")
            expected_range = rule.get("range", "")

            subj_type = graph.nodes[subj].get("type", "") if subj in graph.nodes else ""
            obj_type = graph.nodes[obj].get("type", "") if obj in graph.nodes else ""

            if expected_domain and subj_type and subj_type != expected_domain:
                violations.append({
                    "triple": f"({subj} -{pred}-> {obj})",
                    "issue": f"Subject type '{subj_type}' ≠ expected domain '{expected_domain}'",
                })
            if expected_range and obj_type and obj_type != expected_range:
                violations.append({
                    "triple": f"({subj} -{pred}-> {obj})",
                    "issue": f"Object type '{obj_type}' ≠ expected range '{expected_range}'",
                })

        compliance_rate = 1.0 - (len(violations) / len(triples))
        health = round(compliance_rate * 100, 1)

        return {
            "compliance_rate": round(compliance_rate, 4),
            "violation_count": len(violations),
            "violations": violations[:10],  # cap for readability
            "health_score": health,
            "flag": "red" if compliance_rate < 0.8 else ("yellow" if compliance_rate < 0.95 else "green"),
            "recommendation": (
                f"{len(violations)} schema violations found — check ontology rules."
                if violations else ""
            ),
        }

    def _entity_duplication(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        """Detect potential duplicate entities using name similarity.

        Uses a lightweight character n-gram Jaccard approach (no embedding model needed).
        For production, replace with sentence-transformers cosine similarity.
        """
        if len(entities) < 2:
            return {
                "duplicate_pairs": [],
                "duplicate_entity_count": 0,
                "health_score": 100,
                "flag": "green",
            }

        names = [(e.get("name", ""), e.get("type", "")) for e in entities]
        duplicates: list[dict[str, Any]] = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = self._ngram_jaccard(names[i][0], names[j][0])
                if sim >= self.entity_dedup_threshold:
                    duplicates.append({
                        "entity_a": names[i][0],
                        "entity_b": names[j][0],
                        "similarity": round(sim, 4),
                    })

        dup_ratio = len(duplicates) / max(len(entities), 1)
        health = max(0, 100 - dup_ratio * 500)

        return {
            "duplicate_pairs": duplicates[:20],
            "duplicate_pair_count": len(duplicates),
            "duplicate_entity_count": len(set(
                d["entity_a"] for d in duplicates
            ) | set(d["entity_b"] for d in duplicates)),
            "health_score": round(health, 1),
            "flag": "red" if len(duplicates) > len(entities) * 0.1 else ("yellow" if duplicates else "green"),
            "recommendation": (
                f"{len(duplicates)} potential duplicate pairs — consider entity resolution."
                if duplicates else ""
            ),
        }

    @staticmethod
    def _multi_hop_connectivity(graph: nx.DiGraph) -> dict[str, Any]:
        """Measure how many nodes are reachable in 2-3 hops (needed for chain reasoning)."""
        n = graph.number_of_nodes()
        if n < 3:
            return {
                "reachable_2hop_pct": 0.0,
                "reachable_3hop_pct": 0.0,
                "avg_path_length": 0.0,
                "health_score": 100,
                "flag": "green",
            }

        # Sample to keep computation fast for large graphs
        sample_nodes = list(graph.nodes())[:200]
        hop2_reachable = 0
        hop3_reachable = 0
        total_pairs = 0

        for node in sample_nodes:
            # BFS limited to 2 and 3 hops
            visited_2 = set()
            visited_3 = set()
            queue = [(node, 0)]

            while queue:
                current, depth = queue.pop(0)
                if depth > 3:
                    break
                for neighbor in graph.successors(current):
                    if neighbor not in visited_2 and neighbor not in visited_3:
                        if depth + 1 <= 2:
                            visited_2.add(neighbor)
                        if depth + 1 <= 3:
                            visited_3.add(neighbor)
                        queue.append((neighbor, depth + 1))

            hop2_reachable += len(visited_2)
            hop3_reachable += len(visited_3)
            total_pairs += len(sample_nodes) - 1  # rough upper bound

        pct_2hop = hop2_reachable / max(total_pairs, 1)
        pct_3hop = hop3_reachable / max(total_pairs, 1)

        # Health: more reachable nodes in 2-3 hops = better for multi-hop QA
        avg_reachability = (pct_2hop + pct_3hop) / 2
        health = min(100, avg_reachability * 200)

        return {
            "reachable_2hop_pct": round(pct_2hop, 4),
            "reachable_3hop_pct": round(pct_3hop, 4),
            "health_score": round(health, 1),
            "flag": "red" if avg_reachability < 0.1 else ("yellow" if avg_reachability < 0.3 else "green"),
            "recommendation": (
                "Low multi-hop connectivity — the KG may not support chain reasoning questions."
                if avg_reachability < 0.1
                else ""
            ),
        }

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
        """Character n-gram Jaccard similarity for lightweight dedup detection."""
        if not a or not b:
            return 0.0
        a = a.lower().strip()
        b = b.lower().strip()
        if a == b:
            return 1.0
        ngrams_a = {a[i:i+n] for i in range(len(a) - n + 1)}
        ngrams_b = {b[i:i+n] for i in range(len(b) - n + 1)}
        if not ngrams_a or not ngrams_b:
            return 0.0
        intersection = ngrams_a & ngrams_b
        union = ngrams_a | ngrams_b
        return len(intersection) / len(union)

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 80:
            return "✅ Healthy — KG is structurally sound for SFT data generation."
        elif score >= 60:
            return "⚠️ Fair — some issues detected; review flagged areas before generating SFT data."
        elif score >= 40:
            return "🔶 Concerning — multiple structural issues; SFT data quality may suffer."
        else:
            return "🔴 Poor — significant structural problems; fix before using for LLM training."


def load_kg_for_audit(path: Path) -> tuple[nx.DiGraph, list[dict[str, Any]], list[tuple[str, str, str, str]]]:
    """Load a KG from a JSON export file for auditing."""
    with open(path) as f:
        data = json.load(f)

    entities = data.get("entities", [])
    triples_raw = data.get("triples", [])
    graph = nx.node_link_graph(data.get("graph", {}))

    triples: list[tuple[str, str, str, str]] = []
    for t in triples_raw:
        if isinstance(t, (list, tuple)):
            if len(t) >= 4:
                triples.append((str(t[0]), str(t[1]), str(t[2]), str(t[3])))
            elif len(t) == 3:
                triples.append((str(t[0]), str(t[1]), str(t[2]), ""))

    return graph, entities, triples
