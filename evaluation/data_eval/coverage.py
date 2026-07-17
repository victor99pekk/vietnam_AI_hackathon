"""
Method 1, Step 1.4 — Source Document Fact Coverage

Extracts factual statements from the original source documents and checks
how many of them are represented in the knowledge graph.

This is a *recall* metric: of all the factual content in the source documents,
what fraction did the KG capture?

Two extraction modes (heuristic implemented; LLM mode stub for future):
  - Heuristic (regex): No API needed, covers common factual patterns
  - LLM (future): Uses DeepSeek API for comprehensive fact extraction
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class FactExtractor:
    """Extracts atomic facts from raw text documents.

    Supports two modes:
      - heuristic (regex): No API needed — fast, covers common patterns
      - llm (future): Uses DeepSeek API for comprehensive extraction
    """

    # ── Regex patterns: (regex, predicate_name, evidence_template) ──
    PATTERNS: list[tuple[str, str, str]] = [
        (r'(.+?)\s+(?:was\s+)?born\s+in\s+(.+?)(?:,|\.|$)', "born_in", "born in"),
        (r'(.+?)\s+(?:worked|works)\s+(?:at|for|in)\s+(.+?)(?:,|\.|$)', "worked_at", "worked at"),
        (r'(.+?)\s+(?:studied|studies)\s+(?:at|in)\s+(.+?)(?:,|\.|$)', "studied_at", "studied at"),
        (r'(.+?)\s+(?:died|dies)\s+in\s+(.+?)(?:,|\.|$)', "died_in", "died in"),
        (r'(.+?)\s+(?:discovered|invented|created|developed|wrote|published)\s+(.+?)(?:,|\.|$)', "created", "created"),
        (r'(.+?)\s+is\s+(?:a|an|the)\s+(.+?)(?:,|\.|$)', "is_a", "is a"),
        (r'(.+?)\s+(?:earned|received|got)\s+(?:his|her|a|an)?\s*(.+?)(?:from|at|,|\.|$)', "earned", "earned"),
        (r'(.+?)\s+(?:located|situated)\s+in\s+(.+?)(?:,|\.|$)', "located_in", "located in"),
        (r'(.+?)\s+(?:founded|established)\s+(.+?)(?:,|\.|$)', "founded", "founded"),
        (r'(.+?)\s+(?:married|wed)\s+(.+?)(?:,|\.|$)', "married", "married"),
    ]

    def __init__(
        self,
        mode: str = "heuristic",
        seed: int = 42,
    ) -> None:
        self.mode = mode
        self.seed = seed
        random.seed(seed)

    def extract_from_documents(
        self, documents: list[dict[str, str]], sample_size: int = 30
    ) -> list[dict[str, Any]]:
        """Extract facts from a list of documents.

        Args:
            documents: list of {"content": str, "source": str} dicts
            sample_size: max number of unique facts to return (random sample)

        Returns list of facts with metadata:
            {"subject": str, "predicate": str, "object": str,
             "source": str, "evidence": str}
        """
        all_facts: list[dict[str, Any]] = []

        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")

            if self.mode == "llm":
                # LLM mode not yet implemented — fall back to heuristic
                logger.info("LLM mode not implemented — using heuristic extraction")
                facts = self._extract_heuristic(content, source)
            else:
                facts = self._extract_heuristic(content, source)

            all_facts.extend(facts)

        # Deduplicate by (subject_lower, predicate_lower, object_lower) key
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, Any]] = []
        for f in all_facts:
            key = (
                f["subject"].lower().strip(),
                f["predicate"].lower().strip(),
                f["object"].lower().strip(),
            )
            if key not in seen:
                seen.add(key)
                unique.append(f)

        logger.info(
            "Extracted %d unique facts from %d documents (raw: %d)",
            len(unique), len(documents), len(all_facts),
        )

        # Sample if too many
        if len(unique) > sample_size:
            unique = random.sample(unique, sample_size)
            logger.info("Sampled down to %d facts", sample_size)

        return unique

    # ── Heuristic extraction ──────────────────────────────────

    def _extract_heuristic(self, text: str, source: str) -> list[dict[str, Any]]:
        """Regex-based fact extraction — works offline, no API needed.

        Looks for common factual patterns:
          X was born in Y, X worked at Y, X studied at Y,
          X died in Y, X discovered/invented/created Y,
          X is a/an/the Y, X earned/received Y,
          X located in Y, X founded Y, X married Y
        """
        facts: list[dict[str, Any]] = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue

            for pattern, predicate, _evidence_label in self.PATTERNS:
                match = re.search(pattern, sent, re.IGNORECASE)
                if not match:
                    continue

                subject = match.group(1).strip().rstrip(".,;:!?")
                obj = match.group(2).strip().rstrip(".,;:!?")

                # Basic quality filter: skip if subject or object is too short/empty
                if len(subject) < 2 or len(obj) < 2:
                    continue

                # Skip if both look like generic phrases (not named entities)
                if not self._looks_like_entity(subject) and not self._looks_like_entity(obj):
                    continue

                # Skip overly long strings (likely full clauses, not entities)
                if len(subject) > 120 or len(obj) > 120:
                    continue

                facts.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "source": source,
                    "evidence": sent[:300],
                })

        return facts

    @staticmethod
    def _looks_like_entity(text: str) -> bool:
        """Heuristic: does this text look like a named entity?

        Returns True if the text contains at least one capitalized word
        longer than 2 characters (not at sentence start is assumed).
        """
        words = text.split()
        for w in words:
            if w[0].isupper() and len(w) > 2:
                return True
        return False


class CoverageEvaluator:
    """Evaluates how many source-document facts are captured by the KG.

    Three levels of matching:
      - exact_match: (subject, predicate, object) all match verbatim
      - entity_pair_match: subject & object exist in KG and are connected
      - entity_mention_match: at minimum the subject entity exists in KG
    """

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        extracted_facts: list[dict[str, Any]],
        graph: nx.DiGraph,
        kg_triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Compare extracted facts against the KG and compute coverage scores."""
        if not extracted_facts:
            return {
                "error": "No facts extracted",
                "total_facts": 0,
                "coverage_score": 0.0,
                "health_score": 0.0,
                "verdict": "No facts to evaluate.",
            }

        # ── Build KG lookup structures ────────────────────────
        kg_nodes_lower: dict[str, str] = {
            n.lower(): n for n in graph.nodes()
        }

        kg_triple_keys: set[tuple[str, str, str]] = {
            (s.lower().strip(), p.lower().strip(), o.lower().strip())
            for s, p, o, _ in kg_triples
        }

        # Adjacency: (subject_lower, object_lower) → True if any edge exists
        kg_adjacency: set[tuple[str, str]] = set()
        for s, _p, o, _src in kg_triples:
            kg_adjacency.add((s.lower().strip(), o.lower().strip()))

        # ── Match each extracted fact ─────────────────────────
        per_fact: list[dict[str, Any]] = []
        exact_matches = 0
        pair_matches = 0
        mention_matches = 0
        total = len(extracted_facts)

        for fact in extracted_facts:
            subj_raw = fact["subject"]
            pred_raw = fact["predicate"]
            obj_raw = fact["object"]

            subj = subj_raw.lower().strip()
            pred = pred_raw.lower().strip()
            obj = obj_raw.lower().strip()

            # 1. Exact triple match
            exact = (subj, pred, obj) in kg_triple_keys

            # 2. Entity pair match: both in KG + connected (any edge)
            subj_in_kg = subj in kg_nodes_lower
            obj_in_kg = obj in kg_nodes_lower
            connected = (subj, obj) in kg_adjacency if subj_in_kg and obj_in_kg else False
            pair_match = subj_in_kg and obj_in_kg and connected

            # 3. Entity mention match: at minimum subject in KG
            mention_match = subj_in_kg

            if exact:
                exact_matches += 1
            if pair_match:
                pair_matches += 1
            if mention_match:
                mention_matches += 1

            per_fact.append({
                "fact": f"({subj_raw} | {pred_raw} | {obj_raw})",
                "source": fact["source"],
                "evidence": fact.get("evidence", "")[:200],
                "exact_match": exact,
                "pair_match": pair_match,
                "mention_match": mention_match,
                "subj_in_kg": subj_in_kg,
                "obj_in_kg": obj_in_kg,
                "connected": connected,
            })

        # ── Compute scores ────────────────────────────────────
        exact_rate = exact_matches / total
        pair_rate = pair_matches / total
        mention_rate = mention_matches / total

        # Weighted overall: exact = 50%, pair = 30%, mention = 20%
        overall = (exact_rate * 0.5) + (pair_rate * 0.3) + (mention_rate * 0.2)
        health_score = round(overall * 100, 1)

        # Separate missed facts (no mention match at all)
        missed = [r for r in per_fact if not r["mention_match"]]

        return {
            "total_facts": total,
            "exact_match_count": exact_matches,
            "exact_match_rate": round(exact_rate, 4),
            "pair_match_count": pair_matches,
            "pair_match_rate": round(pair_rate, 4),
            "mention_match_count": mention_matches,
            "mention_match_rate": round(mention_rate, 4),
            "coverage_score": round(overall, 4),
            "health_score": health_score,
            "verdict": self._interpret(overall),
            "missed_facts": missed[:20],
            "per_fact_results": per_fact,
        }

    @staticmethod
    def _interpret(score: float) -> str:
        """Human-readable interpretation of the coverage score."""
        if score >= 0.8:
            return "✅ Excellent — KG captures most source-document facts."
        elif score >= 0.6:
            return "⚠️ Good — KG captures many facts, but some are missing."
        elif score >= 0.4:
            return "🔶 Fair — significant factual gaps; review extraction pipeline."
        else:
            return "🔴 Poor — KG misses many source facts; extraction/triple generation may need tuning."


def compute_kg_coverage(
    kg_path: Path,
    document_paths: list[Path],
    extraction_mode: str = "heuristic",
    fact_sample_size: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Main entry point: compute fact coverage of a KG against source documents.

    Args:
        kg_path: path to knowledge_graph.json
        document_paths: list of source document files/directories
        extraction_mode: "heuristic" (default) or "llm" (future)
        fact_sample_size: max number of unique facts to sample
        seed: random seed for reproducibility

    Returns coverage report dict with health_score, verdict, and per-fact details.
    """
    from evaluation.model_eval.dataset_gen import load_kg, load_raw_documents

    # 1. Load KG
    graph, entities, triples = load_kg(kg_path)
    logger.info(
        "Loaded KG: %d nodes, %d edges, %d triples",
        graph.number_of_nodes(), graph.number_of_edges(), len(triples),
    )

    # 2. Load source documents
    documents = load_raw_documents(document_paths)
    logger.info("Loaded %d source documents", len(documents))

    if not documents:
        return {
            "error": "No source documents found",
            "coverage_score": 0.0,
            "health_score": 0.0,
            "verdict": "No source documents to evaluate.",
        }

    # 3. Extract facts from documents
    extractor = FactExtractor(mode=extraction_mode, seed=seed)
    facts = extractor.extract_from_documents(documents, sample_size=fact_sample_size)

    if not facts:
        return {
            "error": "Could not extract any facts from documents",
            "coverage_score": 0.0,
            "health_score": 0.0,
            "verdict": "No facts could be extracted from documents.",
        }

    # 4. Evaluate coverage
    evaluator = CoverageEvaluator()
    report = evaluator.evaluate(facts, graph, triples)

    return report
