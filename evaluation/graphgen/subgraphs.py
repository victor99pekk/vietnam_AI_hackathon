"""Deterministic GraphGen-style k-hop subgraph organization.

The paper uses comprehension loss to rank candidate edges. This implementation
supports that field when it is available and records a stable-ID fallback when
it is not. It never invents a loss value.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_STRUCTURAL_PREDICATES = {"MENTIONS", "PART_OF", "NEXT"}


def estimate_tokens(text: str) -> int:
    """Return a Unicode-safe, model-independent token estimate.

    GraphGen used the Qwen tokenizer. Keeping the estimator explicit in every
    artifact avoids presenting this lightweight estimate as an exact model
    token count.
    """

    return len(_TOKEN_RE.findall(text or ""))


def _stable_id(prefix: str, *parts: str) -> str:
    material = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(material).hexdigest()[:20]}"


def _as_chunk_ids(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(sorted({str(item) for item in value if item}))
    return (str(value),)


@dataclass(frozen=True)
class KnowledgeEdge:
    id: str
    source: str
    target: str
    description: str
    source_chunk_ids: tuple[str, ...] = ()
    loss: float | None = None

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.description)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "description": self.description,
            "source_chunk_ids": list(self.source_chunk_ids),
            "comprehension_loss": self.loss,
        }


@dataclass
class SamplingResult:
    subgraphs: list[dict[str, Any]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)

    def write(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        subgraphs_path = output_dir / "subgraphs.jsonl"
        audit_path = output_dir / "sampling_audit.jsonl"

        with open(subgraphs_path, "w", encoding="utf-8") as handle:
            for subgraph in self.subgraphs:
                handle.write(json.dumps(subgraph, ensure_ascii=False) + "\n")

        with open(audit_path, "w", encoding="utf-8") as handle:
            for event in self.audit:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        return subgraphs_path, audit_path


def load_graphgen_kg(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[KnowledgeEdge]]:
    """Load entity nodes and descriptive knowledge edges from a KG JSON export.

    Both the current dictionary triple schema and the older tuple schema are
    accepted. Structural edges are excluded because they describe storage and
    provenance, not domain knowledge.
    """

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    raw_nodes = data.get("graph", {}).get("nodes", [])
    nodes = {
        str(node.get("id")): {
            "id": str(node.get("id")),
            "name": str(node.get("name", node.get("id", ""))),
            "type": str(node.get("type", "ENTITY")),
            "description": str(node.get("description", "")),
        }
        for node in raw_nodes
        if node.get("id") and node.get("type") not in {"Chunk", "Document"}
    }

    edges: list[KnowledgeEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in data.get("triples", []):
        if isinstance(raw, dict):
            source = str(raw.get("subject", ""))
            predicate = str(raw.get("predicate", ""))
            target = str(raw.get("object", ""))
            description = str(
                raw.get("description") or raw.get("evidence_sentence") or ""
            ).strip()
            chunk_ids = _as_chunk_ids(
                raw.get("source_chunk_ids") or raw.get("source_chunk_id")
            )
            loss_value = raw.get("comprehension_loss", raw.get("loss"))
        elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
            source, predicate, target = map(str, raw[:3])
            description = str(raw[5] if len(raw) > 5 else raw[3] if len(raw) > 3 else "").strip()
            chunk_ids = _as_chunk_ids(raw[4] if len(raw) > 4 else None)
            loss_value = None
        else:
            continue

        if (
            not source
            or not target
            or predicate.upper() in _STRUCTURAL_PREDICATES
            or source not in nodes
            or target not in nodes
        ):
            continue

        signature = (source, target, description)
        if signature in seen:
            continue
        seen.add(signature)

        try:
            loss = float(loss_value) if loss_value is not None else None
        except (TypeError, ValueError):
            loss = None

        edge_id = _stable_id("edge", source, target, description)
        edges.append(
            KnowledgeEdge(
                id=edge_id,
                source=source,
                target=target,
                description=description,
                source_chunk_ids=chunk_ids,
                loss=loss,
            )
        )

    return nodes, sorted(edges, key=lambda edge: edge.id)


class GraphGenSubgraphSampler:
    """Organize descriptive KG edges into bounded, reproducible subgraphs."""

    method = "graphgen-khop-v1"
    token_estimator = "unicode_regex_estimate"

    def __init__(
        self,
        *,
        max_depth: int = 2,
        max_premise_tokens: int = 256,
        max_extra_edges: int = 5,
        edge_sampling: str = "max_loss",
        bidirectional: bool = True,
        seed: int = 42,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if max_premise_tokens <= 0:
            raise ValueError("max_premise_tokens must be > 0")
        if max_extra_edges < 0:
            raise ValueError("max_extra_edges must be >= 0")
        if edge_sampling not in {"max_loss", "min_loss", "random"}:
            raise ValueError("edge_sampling must be max_loss, min_loss, or random")

        self.max_depth = max_depth
        self.max_premise_tokens = max_premise_tokens
        self.max_extra_edges = max_extra_edges
        self.edge_sampling = edge_sampling
        self.bidirectional = bidirectional
        self.seed = seed

    def sample(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[KnowledgeEdge],
        *,
        max_subgraphs: int | None = None,
    ) -> SamplingResult:
        edge_by_id = {edge.id: edge for edge in edges}
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for edge in edges:
            adjacency.setdefault(edge.source, set()).add(edge.id)
            if self.bidirectional:
                adjacency.setdefault(edge.target, set()).add(edge.id)

        seed_edges = self._rank_edges(edges, seed_edge_id="seeds")
        if max_subgraphs is not None:
            seed_edges = seed_edges[:max_subgraphs]

        result = SamplingResult()
        for seed_edge in seed_edges:
            subgraph, events = self._expand_seed(
                seed_edge, nodes, edge_by_id, adjacency
            )
            result.audit.extend(events)
            if subgraph is not None:
                result.subgraphs.append(subgraph)

        result.subgraphs.sort(key=lambda item: item["id"])
        result.audit.sort(
            key=lambda item: (item["seed_edge_id"], item["candidate_edge_id"])
        )
        return result

    def _expand_seed(
        self,
        seed_edge: KnowledgeEdge,
        nodes: dict[str, dict[str, Any]],
        edge_by_id: dict[str, KnowledgeEdge],
        adjacency: dict[str, set[str]],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        selected_edge_ids = {seed_edge.id}
        selected_node_ids = {seed_edge.source, seed_edge.target}
        node_depth = {seed_edge.source: 0, seed_edge.target: 0}
        excluded: set[str] = set()
        events: list[dict[str, Any]] = []
        premise_tokens = self._premise_tokens(
            selected_node_ids, selected_edge_ids, nodes, edge_by_id
        )

        if premise_tokens > self.max_premise_tokens:
            events.append(
                self._audit_event(
                    seed_edge, seed_edge, "rejected", "seed_exceeds_token_budget", 0,
                    premise_tokens,
                )
            )
            return None, events

        events.append(
            self._audit_event(
                seed_edge, seed_edge, "included", "seed_edge", 0, premise_tokens
            )
        )

        while len(selected_edge_ids) - 1 < self.max_extra_edges:
            candidates: list[tuple[KnowledgeEdge, int]] = []
            for node_id in sorted(selected_node_ids):
                for edge_id in sorted(adjacency.get(node_id, set())):
                    if edge_id in selected_edge_ids or edge_id in excluded:
                        continue
                    edge = edge_by_id[edge_id]
                    connected_depths = [
                        node_depth[endpoint]
                        for endpoint in (edge.source, edge.target)
                        if endpoint in node_depth
                    ]
                    if not connected_depths:
                        continue
                    expansion_depth = min(connected_depths) + 1
                    candidates.append((edge, expansion_depth))

            if not candidates:
                break

            unique_candidates = {edge.id: (edge, depth) for edge, depth in candidates}
            ranked = self._rank_edges(
                [item[0] for item in unique_candidates.values()], seed_edge.id
            )
            candidate = ranked[0]
            expansion_depth = unique_candidates[candidate.id][1]

            if expansion_depth > self.max_depth:
                excluded.add(candidate.id)
                events.append(
                    self._audit_event(
                        seed_edge, candidate, "rejected", "depth_limit",
                        expansion_depth, premise_tokens,
                    )
                )
                continue

            proposed_edges = selected_edge_ids | {candidate.id}
            proposed_nodes = selected_node_ids | {candidate.source, candidate.target}
            proposed_tokens = self._premise_tokens(
                proposed_nodes, proposed_edges, nodes, edge_by_id
            )
            if proposed_tokens > self.max_premise_tokens:
                excluded.add(candidate.id)
                events.append(
                    self._audit_event(
                        seed_edge, candidate, "rejected", "token_budget",
                        expansion_depth, proposed_tokens,
                    )
                )
                continue

            selected_edge_ids = proposed_edges
            selected_node_ids = proposed_nodes
            premise_tokens = proposed_tokens
            for endpoint in (candidate.source, candidate.target):
                if endpoint not in node_depth:
                    node_depth[endpoint] = expansion_depth
                else:
                    node_depth[endpoint] = min(node_depth[endpoint], expansion_depth)
            events.append(
                self._audit_event(
                    seed_edge, candidate, "included", self._selection_reason(candidate, edges=edge_by_id.values()),
                    expansion_depth, premise_tokens,
                )
            )

        remaining = {
            edge_id
            for node_id in selected_node_ids
            for edge_id in adjacency.get(node_id, set())
            if edge_id not in selected_edge_ids and edge_id not in excluded
        }
        for edge_id in sorted(remaining):
            events.append(
                self._audit_event(
                    seed_edge, edge_by_id[edge_id], "rejected", "edge_limit",
                    self.max_depth + 1, premise_tokens,
                )
            )

        ordered_edges = [edge_by_id[edge_id] for edge_id in sorted(selected_edge_ids)]
        ordered_nodes = [nodes[node_id] for node_id in sorted(selected_node_ids)]
        subgraph_id = _stable_id(
            "subgraph",
            seed_edge.id,
            *sorted(selected_edge_ids),
            str(self.max_depth),
            str(self.max_premise_tokens),
        )
        subgraph = {
            "id": subgraph_id,
            "method": self.method,
            "seed_edge_id": seed_edge.id,
            "edge_sampling": self.edge_sampling,
            "selection_basis": self._selection_basis(ordered_edges),
            "max_depth": self.max_depth,
            "max_premise_tokens": self.max_premise_tokens,
            "premise_tokens": premise_tokens,
            "token_estimator": self.token_estimator,
            "nodes": ordered_nodes,
            "edges": [edge.to_dict() for edge in ordered_edges],
            "source_chunk_ids": sorted(
                {chunk_id for edge in ordered_edges for chunk_id in edge.source_chunk_ids}
            ),
        }
        return subgraph, events

    def _rank_edges(
        self, edges: Iterable[KnowledgeEdge], seed_edge_id: str
    ) -> list[KnowledgeEdge]:
        edges = list(edges)
        if self.edge_sampling == "random":
            return sorted(
                edges,
                key=lambda edge: hashlib.sha256(
                    f"{self.seed}:{seed_edge_id}:{edge.id}".encode("utf-8")
                ).hexdigest(),
            )
        if self.edge_sampling == "min_loss":
            return sorted(
                edges,
                key=lambda edge: (
                    edge.loss is None,
                    edge.loss if edge.loss is not None else float("inf"),
                    edge.id,
                ),
            )
        return sorted(
            edges,
            key=lambda edge: (
                edge.loss is None,
                -(edge.loss if edge.loss is not None else 0.0),
                edge.id,
            ),
        )

    def _selection_basis(self, edges: Iterable[KnowledgeEdge]) -> str:
        if self.edge_sampling == "random":
            return "seeded_random"
        if any(edge.loss is not None for edge in edges):
            return "comprehension_loss"
        return "stable_id_fallback_no_comprehension_loss"

    def _selection_reason(
        self, edge: KnowledgeEdge, *, edges: Iterable[KnowledgeEdge]
    ) -> str:
        if self.edge_sampling == "random":
            return "seeded_random"
        if edge.loss is not None or any(item.loss is not None for item in edges):
            return self.edge_sampling
        return "stable_id_fallback"

    @staticmethod
    def _premise_tokens(
        node_ids: set[str],
        edge_ids: set[str],
        nodes: dict[str, dict[str, Any]],
        edges: dict[str, KnowledgeEdge],
    ) -> int:
        node_text = " ".join(
            f"{nodes[node_id].get('name', '')} {nodes[node_id].get('description', '')}"
            for node_id in sorted(node_ids)
        )
        edge_text = " ".join(edges[edge_id].description for edge_id in sorted(edge_ids))
        return estimate_tokens(f"{node_text} {edge_text}")

    def _audit_event(
        self,
        seed: KnowledgeEdge,
        candidate: KnowledgeEdge,
        decision: str,
        reason: str,
        expansion_depth: int,
        premise_tokens: int,
    ) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed_edge_id": seed.id,
            "candidate_edge_id": candidate.id,
            "decision": decision,
            "reason": reason,
            "expansion_depth": expansion_depth,
            "premise_tokens_after_decision": premise_tokens,
            "max_premise_tokens": self.max_premise_tokens,
            "comprehension_loss": candidate.loss,
            "source_chunk_ids": list(candidate.source_chunk_ids),
        }
