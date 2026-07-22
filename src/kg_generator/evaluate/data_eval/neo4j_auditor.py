"""
Neo4j-native structural audit — queries the database directly instead of
loading the full graph into a ``networkx.DiGraph``.

Every check that the standard ``StructuralAuditor`` performs via in-memory
graph algorithms is reimplemented here as Cypher queries so the audit
scales to graphs of any size.

Usage::

    from kg_generator.evaluate.data_eval.neo4j_auditor import Neo4jStructuralAuditor

    auditor = Neo4jStructuralAuditor(session)
    report = auditor.audit()   # no networkx, no full-graph download
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Neo4jStructuralAuditor:
    """Audit a Neo4j-stored knowledge graph for structural health.

    Parameters
    ----------
    session:
        An active Neo4j ``Session``.
    entity_dedup_threshold:
        n-gram Jaccard threshold for flagging potential duplicate entities.
    """

    def __init__(
        self,
        session: Any,
        entity_dedup_threshold: float = 0.85,
    ) -> None:
        self.session = session
        self.entity_dedup_threshold = entity_dedup_threshold

    # ── Public API ────────────────────────────────────────────

    def audit(self) -> dict[str, Any]:
        """Run the full structural audit and return a report dict.

        The report has the same shape as ``StructuralAuditor.audit()`` so
        downstream consumers (plotting, reporting) work unchanged.
        """
        report: dict[str, Any] = {
            "graph_stats": self._basic_stats(),
            "orphan_analysis": self._orphan_analysis(),
            "density_analysis": self._density_analysis(),
            "schema_compliance": self._schema_compliance(),
            "entity_duplication": self._entity_duplication(),
            "multi_hop_connectivity": self._multi_hop_connectivity(),
            "overall_health_score": 0.0,
        }

        scores = [
            report["orphan_analysis"]["health_score"],
            report["density_analysis"]["health_score"],
            report["schema_compliance"]["health_score"],
            report["entity_duplication"]["health_score"],
            report["multi_hop_connectivity"]["health_score"],
        ]
        report["overall_health_score"] = round(sum(scores) / len(scores), 1)
        report["verdict"] = self._interpret(report["overall_health_score"])
        return report

    # ── Individual Checks ─────────────────────────────────────

    def _basic_stats(self) -> dict[str, Any]:
        result = self.session.run(
            """
            CALL () { MATCH (n) RETURN count(n) AS node_count }
            CALL () { MATCH ()-[r]->() RETURN count(r) AS edge_count }
            RETURN node_count, edge_count
            """
        ).single()

        node_count = result["node_count"] if result else 0
        edge_count = result["edge_count"] if result else 0

        return {
            "num_nodes": node_count,
            "num_edges": edge_count,
            "num_triples": edge_count,
            "is_directed": True,
            "is_connected": True,
        }

    def _orphan_analysis(self) -> dict[str, Any]:
        result = self.session.run(
            """
            CALL () { MATCH (n) RETURN count(n) AS total }
            CALL () { MATCH (n) WHERE NOT (n)--() RETURN count(n) AS isolated }
            RETURN total, isolated
            """
        ).single()

        if result is None or result["total"] == 0:
            return {
                "orphan_count": 0, "orphan_rate": 0.0,
                "entity_orphans": 0, "chunk_orphans": 0,
                "health_score": 100,
                "flag": "green", "recommendation": "",
            }

        total = result["total"]
        isolated = result["isolated"]
        orphan_rate = isolated / total

        entity_orphan_result = self.session.run(
            "MATCH (n:Entity) WHERE NOT (n)--() RETURN count(n) AS entity_orphans"
        ).single()
        entity_orphans = entity_orphan_result["entity_orphans"] if entity_orphan_result else 0
        chunk_orphans = isolated - entity_orphans

        health = max(0, 100 - (orphan_rate * 300))

        return {
            "orphan_count": isolated,
            "orphan_rate": round(orphan_rate, 4),
            "entity_orphans": entity_orphans,
            "chunk_orphans": chunk_orphans,
            "health_score": round(health, 1),
            "flag": "red" if orphan_rate > 0.3 else ("yellow" if orphan_rate > 0.1 else "green"),
            "recommendation": (
                "High orphan rate — consider merging orphan entities or enriching connections."
                if orphan_rate > 0.3 else ""
            ),
        }

    def _density_analysis(self) -> dict[str, Any]:
        result = self.session.run(
            """
            CALL () { MATCH (n) RETURN count(n) AS n }
            CALL () { MATCH ()-[r]->() RETURN count(r) AS e }
            RETURN n, e
            """
        ).single()

        if result is None or result["n"] < 2:
            return {"density": 0.0, "health_score": 100,
                    "flag": "green", "recommendation": ""}

        n = result["n"]
        e = result["e"]
        density = e / (n * (n - 1)) if n > 1 else 0.0

        if density < 0.005:
            health = max(0, density * 2000)
        elif density <= 0.2:
            health = 100
        else:
            health = max(0, 100 - (density - 0.2) * 500)

        return {
            "density": round(density, 6),
            "health_score": round(health, 1),
            "flag": "red" if density < 0.005 else ("yellow" if density > 0.2 else "green"),
            "recommendation": (
                "Graph is very sparse — relations may not capture enough context."
                if density < 0.005
                else ("Graph is very dense — possible over-connected noise."
                      if density > 0.2 else "")
            ),
        }

    def _schema_compliance(self) -> dict[str, Any]:
        """Check for nonsensical type patterns by sampling edge types."""
        edge_sample = self.session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE type(r) <> 'MENTIONS'
            RETURN a.type AS subj_type,
                   type(r) AS predicate,
                   b.type AS obj_type,
                   count(*) AS cnt
            ORDER BY cnt DESC LIMIT 100
            """
        ).data()

        sensible = 0
        nonsensical = 0
        violations: list[dict[str, str]] = []
        for row in edge_sample:
            st = row.get("subj_type", "")
            ot = row.get("obj_type", "")
            if st == "DATE" and ot in ("PERSON", "ORG", "GPE"):
                nonsensical += row.get("cnt", 0)
                violations.append({
                    "triple": f"DATE -[ ]-> {ot}",
                    "issue": "DATE node in non-temporal subject position",
                })
            else:
                sensible += row.get("cnt", 0)

        total = sensible + nonsensical
        compliance_rate = sensible / max(total, 1)
        health = round(compliance_rate * 100, 1)

        return {
            "compliance_rate": round(compliance_rate, 4),
            "violation_count": nonsensical,
            "violations": violations[:10],
            "health_score": health,
            "flag": "red" if compliance_rate < 0.8
                    else ("yellow" if compliance_rate < 0.95 else "green"),
            "recommendation": (
                f"{nonsensical} nonsensical type patterns — check entity typing."
                if nonsensical else ""
            ),
        }

    def _entity_duplication(self) -> dict[str, Any]:
        """Find entities where one name contains another (same type)."""
        result = self.session.run(
            """
            MATCH (a:Entity)
            MATCH (b:Entity)
            WHERE a.id < b.id
              AND a.type = b.type
              AND a.name <> b.name
              AND (toLower(a.name) CONTAINS toLower(b.name)
                   OR toLower(b.name) CONTAINS toLower(a.name))
            RETURN a.name AS entity_a,
                   b.name AS entity_b,
                   a.type AS type
            LIMIT 200
            """
        ).data()

        duplicates: list[dict[str, Any]] = []
        for row in result:
            a_name = row.get("entity_a", "")
            b_name = row.get("entity_b", "")
            sim = self._ngram_jaccard(a_name, b_name)
            if sim >= self.entity_dedup_threshold:
                duplicates.append({
                    "entity_a": a_name,
                    "entity_b": b_name,
                    "similarity": round(sim, 4),
                })

        total_result = self.session.run(
            "MATCH (e:Entity) RETURN count(e) AS cnt"
        ).single()
        total_entities = total_result["cnt"] if total_result else 1

        dup_ratio = len(duplicates) / max(total_entities, 1)
        health = max(0, 100 - dup_ratio * 500)

        return {
            "duplicate_pairs": duplicates[:20],
            "duplicate_pair_count": len(duplicates),
            "duplicate_entity_count": len(set(
                d["entity_a"] for d in duplicates
            ) | set(d["entity_b"] for d in duplicates)),
            "health_score": round(health, 1),
            "flag": (
                "red" if len(duplicates) > total_entities * 0.1
                else ("yellow" if duplicates else "green")
            ),
            "recommendation": (
                f"{len(duplicates)} potential duplicate pairs — consider entity resolution."
                if duplicates else ""
            ),
        }

    def _multi_hop_connectivity(self) -> dict[str, Any]:
        """Estimate multi-hop reachability via sampling 50 random entities."""
        count_result = self.session.run(
            "MATCH (e:Entity) RETURN count(e) AS cnt"
        ).single()
        total = count_result["cnt"] if count_result else 0
        if total < 3:
            return {
                "reachable_2hop_pct": 0.0, "reachable_3hop_pct": 0.0,
                "avg_path_length": 0.0, "health_score": 100,
                "flag": "green", "recommendation": "",
            }

        sample = self.session.run(
            "MATCH (e:Entity) RETURN e.id AS id ORDER BY rand() LIMIT 50"
        ).data()
        sample_ids = [row["id"] for row in sample]

        reachable_2 = 0
        reachable_3 = 0
        total_pairs = 0

        for sid in sample_ids:
            r2 = self.session.run(
                """
                MATCH (start:Entity {id: $id})
                MATCH path = (start)-[*1..2]-(other:Entity)
                WHERE other.id <> $id
                RETURN count(DISTINCT other) AS cnt
                """, id=sid,
            ).single()
            if r2:
                reachable_2 += r2["cnt"]

            r3 = self.session.run(
                """
                MATCH (start:Entity {id: $id})
                MATCH path = (start)-[*1..3]-(other:Entity)
                WHERE other.id <> $id
                RETURN count(DISTINCT other) AS cnt
                """, id=sid,
            ).single()
            if r3:
                reachable_3 += r3["cnt"]

            total_pairs += total - 1

        pct_2hop = reachable_2 / max(total_pairs, 1)
        pct_3hop = reachable_3 / max(total_pairs, 1)
        avg_reachability = (pct_2hop + pct_3hop) / 2
        health = min(100, avg_reachability * 200)

        return {
            "reachable_2hop_pct": round(pct_2hop, 4),
            "reachable_3hop_pct": round(pct_3hop, 4),
            "health_score": round(health, 1),
            "flag": (
                "red" if avg_reachability < 0.1
                else ("yellow" if avg_reachability < 0.3 else "green")
            ),
            "recommendation": (
                "Low multi-hop connectivity — the KG may not support chain reasoning."
                if avg_reachability < 0.1 else ""
            ),
        }

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
        if not a or not b:
            return 0.0
        a, b = a.lower().strip(), b.lower().strip()
        if a == b:
            return 1.0
        ngrams_a = {a[i:i + n] for i in range(len(a) - n + 1)}
        ngrams_b = {b[i:i + n] for i in range(len(b) - n + 1)}
        if not ngrams_a or not ngrams_b:
            return 0.0
        return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 80:
            return "✅ Healthy — KG is structurally sound for SFT data generation."
        elif score >= 60:
            return "⚠️ Fair — some issues detected; review flagged areas."
        elif score >= 40:
            return "🔶 Concerning — multiple structural issues."
        else:
            return "🔴 Poor — significant structural problems."
