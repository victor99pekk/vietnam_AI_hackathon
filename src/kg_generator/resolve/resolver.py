"""Entity resolution — merging duplicate entity mentions into canonical nodes."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolves duplicate entities using embedding similarity or string matching."""

    def __init__(
        self,
        threshold: float = 0.80,
        method: str = "embedding",
    ) -> None:
        self.threshold = threshold
        self.method = method
        self._embedder = None

    @property
    def embedder(self):
        """Lazy-load the sentence transformer model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Multilingual model — covers English + Vietnamese
                self._embedder = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
            except Exception:
                logger.warning(
                    "sentence-transformers unavailable — falling back to string matching"
                )
                return None
        return self._embedder

    def resolve(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge duplicate entities, returning canonical set."""
        if not entities:
            return []

        if self.method == "embedding" and self.embedder is not None:
            return self._embedding_resolve(entities)
        return self._string_resolve(entities)

    def _embedding_resolve(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cluster entities by embedding similarity."""
        names = [e["name"] for e in entities]
        embeddings = self.embedder.encode(names, show_progress_bar=False)

        # Greedy clustering
        clusters: list[list[int]] = []
        assigned: set[int] = set()

        for i in range(len(names)):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, len(names)):
                if j in assigned:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                ))
                if sim >= self.threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        # Merge each cluster into a canonical entity
        resolved: list[dict[str, Any]] = []
        for cluster in clusters:
            canonical = self._merge_cluster(entities, cluster)
            resolved.append(canonical)

        logger.info(f"Entity resolution: {len(entities)} -> {len(resolved)} unique entities")
        return resolved

    def _string_resolve(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve by normalized string similarity (no embedding deps)."""
        resolved: list[dict[str, Any]] = []
        seen: list[str] = []

        for entity in entities:
            name = entity["name"].lower().strip()
            matched = False

            for i, existing in enumerate(seen):
                if self._string_similarity(name, existing) >= self.threshold:
                    # Merge into existing canonical entity
                    canonical = resolved[i]

                    # Combine aliases
                    merged_aliases = set(canonical.get("aliases", []))
                    merged_aliases.update(entity.get("aliases", []))
                    merged_aliases.add(entity.get("name", ""))
                    canonical["aliases"] = sorted(a for a in merged_aliases if a)

                    # Take highest confidence
                    canonical["confidenceScore"] = max(
                        canonical.get("confidenceScore", 0), entity.get("confidenceScore", 0)
                    )

                    # Keep longest description
                    if len(entity.get("description", "")) > len(canonical.get("description", "")):
                        canonical["description"] = entity["description"]

                    # Accumulate sources as a list
                    existing = canonical.get("source", [])
                    if not isinstance(existing, list):
                        existing = [existing] if existing else []
                    new_src = entity.get("source", [])
                    if isinstance(new_src, str):
                        new_src = [new_src] if new_src else []
                    canonical["source"] = list(dict.fromkeys(existing + new_src))

                    matched = True
                    break

            if not matched:
                seen.append(name)
                resolved.append(dict(entity))

        logger.info(f"String resolution: {len(entities)} -> {len(resolved)} unique entities")
        return resolved

    def _merge_cluster(
        self, entities: list[dict[str, Any]], cluster: list[int]
    ) -> dict[str, Any]:
        """Merge a cluster of entity dicts into one canonical entity."""
        cluster_entities = [entities[i] for i in cluster]

        # Best entity by confidence
        best = max(cluster_entities, key=lambda e: (e.get("confidenceScore", 0), len(e.get("name", ""))))

        all_aliases: list[str] = []
        all_sources: list[str] = []
        best_description = ""

        for e in cluster_entities:
            all_aliases.extend(e.get("aliases", []))
            all_aliases.append(e.get("name", ""))
            if e.get("source"):
                src = e["source"]
                if isinstance(src, list):
                    all_sources.extend(src)
                else:
                    all_sources.append(src)
            if len(e.get("description", "")) > len(best_description):
                best_description = e["description"]

        return {
            "id": best.get("id", f"entity:{best['name'].lower().replace(' ', '_')}"),
            "name": best["name"],
            "type": best.get("type", "ENTITY"),
            "aliases": sorted(set(a for a in all_aliases if a)),
            "description": best_description,
            "confidenceScore": max(e.get("confidenceScore", 0) for e in cluster_entities),
            "importanceScore": max(e.get("importanceScore", 0) for e in cluster_entities),
            "source": list(dict.fromkeys(all_sources)),  # deduped, order preserved
            "embedding": best.get("embedding"),
            "updatedAt": max((e.get("updatedAt", "") for e in cluster_entities), default=""),
        }

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        """Simple token-overlap similarity between two strings."""
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
