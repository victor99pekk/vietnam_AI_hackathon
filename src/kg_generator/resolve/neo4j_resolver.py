"""Neo4j-backed entity resolution — resolve new entities against the existing graph.

Instead of clustering all entities in RAM (``EntityResolver``), this resolver
queries Neo4j for candidate matches by name and type, then either merges the
new entity into an existing canonical node or creates a new one.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Neo4jEntityResolver:
    """Resolve entities from a new document against the existing Neo4j graph.

    For each extracted entity this resolver:

    1. Queries Neo4j for existing ``:Entity`` nodes with the same type and a
       similar name (case-insensitive prefix/suffix/contains, or Levenshtein).
    2. If a candidate exceeds the similarity threshold the new entity's aliases
       and confidence score are folded into the existing canonical node.
    3. Otherwise a brand-new ``:Entity`` node is created via ``MERGE``.

    This avoids loading the full entity set into RAM and scales to
    arbitrarily-large graphs.
    """

    def __init__(
        self,
        session: Any,
        threshold: float = 0.85,
    ) -> None:
        """
        Parameters
        ----------
        session:
            An active Neo4j ``Session``.
        threshold:
            String-similarity threshold (0–1) above which two entity names
            are considered the same.
        """
        self.session = session
        self.threshold = threshold
        self._id_map: dict[str, str] = {}  # original_id → canonical_id

    # ── Public API ─────────────────────────────────────────────────

    def resolve(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve a batch of extracted entities against the Neo4j graph.

        Returns
        -------
        list[dict]
            The resolved entities.  Entities that matched an existing node
            carry the canonical ``id`` and merged metadata.
        """
        if not entities:
            return []

        resolved: list[dict[str, Any]] = []
        for entity in entities:
            canonical = self._resolve_one(entity)
            resolved.append(canonical)
        return resolved

    def resolve_with_mapping(
        self, entities: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Resolve and also return a mapping from every original ID → canonical ID."""
        resolved = self.resolve(entities)
        return resolved, self._id_map

    # ── Per-entity logic ───────────────────────────────────────────

    def _resolve_one(self, entity: dict[str, Any]) -> dict[str, Any]:
        """Resolve a single entity: find canonical match or create new."""
        name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("type", entity.get("label", "ENTITY"))).strip()
        original_id = str(entity.get("id", ""))

        if not name:
            self._id_map[original_id] = original_id
            return entity

        # 1. Query Neo4j for candidate matches
        candidates = self._query_candidates(name, entity_type)

        # 2. Score each candidate
        best_match = None
        best_score = 0.0
        for candidate in candidates:
            score = self._string_similarity(name.lower(), candidate["name"].lower())
            if score > best_score:
                best_score = score
                best_match = candidate

        # 3. Merge or create
        if best_match is not None and best_score >= self.threshold:
            canonical = self._merge_into_existing(best_match, entity)
        else:
            canonical = self._create_new(entity)

        # Track the mapping
        canonical_id = str(canonical.get("id", ""))
        if original_id and canonical_id:
            self._id_map[original_id] = canonical_id

        return canonical

    # ── Neo4j queries ──────────────────────────────────────────────

    def _query_candidates(
        self, name: str, entity_type: str
    ) -> list[dict[str, Any]]:
        """Return up to 10 existing Entity nodes with a similar name and same type."""
        result = self.session.run(
            """
            MATCH (e:Entity {type: $type})
            WHERE toLower(e.name) CONTAINS toLower($name)
               OR toLower($name) CONTAINS toLower(e.name)
            RETURN e.id   AS id,
                   e.name AS name,
                   e.confidenceScore AS confidenceScore,
                   e.aliases AS aliases
            LIMIT 10
            """,
            type=entity_type,
            name=name,
        )
        return [dict(record) for record in result]

    def _merge_into_existing(
        self,
        candidate: dict[str, Any],
        entity: dict[str, Any],
    ) -> dict[str, Any]:
        """Fold the new entity into an existing canonical node in Neo4j."""
        canonical_id = str(candidate["id"])
        new_name = str(entity.get("name", ""))
        new_confidence = float(entity.get("confidenceScore", 1.0))

        # Update the existing node in Neo4j
        self.session.run(
            """
            MATCH (e:Entity {id: $id})
            SET e.aliases = CASE
                    WHEN $new_name IN coalesce(e.aliases, [])
                    THEN coalesce(e.aliases, [])
                    ELSE coalesce(e.aliases, []) + [$new_name]
                END,
                e.confidenceScore = CASE
                    WHEN $confidence > coalesce(e.confidenceScore, 0)
                    THEN $confidence
                    ELSE coalesce(e.confidenceScore, 0)
                END
            """,
            id=canonical_id,
            new_name=new_name,
            confidence=new_confidence,
        )

        # Build the resolved dict reflecting the canonical node
        merged_aliases = list(set(
            (candidate.get("aliases") or []) + [new_name]
        ))
        return {
            "id": canonical_id,
            "name": candidate.get("name", new_name),
            "type": entity.get("type", entity.get("label", "ENTITY")),
            "aliases": merged_aliases,
            "confidenceScore": max(
                float(candidate.get("confidenceScore", 0)), new_confidence
            ),
            "description": entity.get("description", ""),
            "source": entity.get("source", []),
            "embedding": entity.get("embedding"),
        }

    def _create_new(self, entity: dict[str, Any]) -> dict[str, Any]:
        """Create a brand-new Entity node in Neo4j and return its dict."""
        entity_id = str(entity.get("id", ""))

        self.session.run(
            """
            MERGE (n:Entity {id: $id})
            SET n.name            = $name,
                n.type            = $type,
                n.description     = $description,
                n.importanceScore = $importanceScore,
                n.confidenceScore = $confidenceScore,
                n.embedding       = $embedding,
                n.aliases         = $aliases
            REMOVE n.entityType
            """,
            id=entity_id,
            name=entity.get("name", ""),
            type=entity.get("type", entity.get("label", "ENTITY")),
            description=entity.get("description", ""),
            importanceScore=entity.get("importanceScore", 0.0),
            confidenceScore=entity.get("confidenceScore", 1.0),
            embedding=entity.get("embedding"),
            aliases=entity.get("aliases", []),
        )

        return dict(entity)

    # ── Similarity helpers ─────────────────────────────────────────

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        """Simple token-overlap similarity between two lower-cased strings."""
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)
