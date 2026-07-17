"""
Method 1, Step 1.4 — Source Document Fact Coverage

Extracts factual statements from the original source documents and checks
how many of them are represented in the knowledge graph.

This is a *recall* metric: of all the factual content in the source documents,
what fraction did the KG capture?

Two extraction modes (heuristic implemented; LLM mode stub for future):
  - Heuristic (regex): No API needed, covers common factual patterns
  - LLM (future): Uses DeepSeek API for comprehensive fact extraction

Entity matching uses normalized comparison + substring fallback for robustness
against minor text differences (e.g. "University of Manchester" vs
"the University of Manchester").
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# ── Stopwords to strip from entity names during normalization ──
_LEADING_STRIP = re.compile(r'^(?:the|a|an)\s+', re.IGNORECASE)
_TRAILING_STRIP = re.compile(r'[,;:.!?\s]+$')
_WHITESPACE = re.compile(r'\s+')


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
        # ── Additional patterns ──
        (r'(.+?)\s+(?:won|received|was\s+awarded)\s+(?:the\s+)?(.+?)(?:,|\.|$)', "won", "won"),
        (r'(.+?)\s+(?:attended|graduated\s+from)\s+(.+?)(?:,|\.|$)', "attended", "attended"),
        (r'(.+?)\s+(?:is\s+)?(?:known|famous)\s+for\s+(.+?)(?:,|\.|$)', "known_for", "known for"),
        (r'(.+?)\s+(?:served|acted)\s+as\s+(?:a|an|the)?\s*(.+?)(?:,|\.|$)', "served_as", "served as"),
    ]

    # ── Pronoun patterns to resolve ──
    _PRONOUNS = {"he", "she", "it", "they", "his", "her", "its", "their"}

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

        Includes pronoun resolution and subject cleaning for better accuracy.
        """
        facts: list[dict[str, Any]] = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Track the last proper-name entity seen per document for pronoun resolution
        last_named_entity: str | None = None

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue

            # Update last_named_entity from this sentence for pronoun resolution
            _named = self._extract_first_named_entity(sent)
            if _named:
                last_named_entity = _named

            for pattern, predicate, _evidence_label in self.PATTERNS:
                match = re.search(pattern, sent, re.IGNORECASE)
                if not match:
                    continue

                subject = match.group(1).strip().rstrip(".,;:!?")
                obj = match.group(2).strip().rstrip(".,;:!?")

                # ── Clean subject (strip leading noise, resolve pronouns) ──
                subject = self._clean_subject(subject)
                subject = self._resolve_pronoun(subject, last_named_entity)

                # ── Quality filters ──
                if len(subject) < 2 or len(obj) < 2:
                    continue
                if not self._looks_like_entity(subject) and not self._looks_like_entity(obj):
                    continue
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

    # ── Subject cleaning helpers ──────────────────────────────

    @staticmethod
    def _clean_subject(raw: str) -> str:
        """Strip leading noise from a regex-extracted subject.

        Examples:
            "During World War II, Turing" → "Turing"
            "the University of Manchester" → "University of Manchester"
            "In 1903, she" → "she"
        """
        # Strip leading adverbial/prepositional phrases ("During X, Y", "In 1903, Z")
        cleaned = re.sub(r'^(?:during|in|after|before|by|at|on|from)\s+.+?,\s*', '', raw, flags=re.IGNORECASE)
        # Strip leading articles
        cleaned = _LEADING_STRIP.sub('', cleaned).strip()
        # Strip trailing noise
        cleaned = _TRAILING_STRIP.sub('', cleaned).strip()
        return cleaned

    @staticmethod
    def _resolve_pronoun(subject: str, last_entity: str | None) -> str:
        """Replace pronoun subjects with the last known proper entity.

        Example: "He" → "Alan Turing" (if last_entity is "Alan Turing")
        """
        if not last_entity:
            return subject

        lower = subject.lower().strip()
        if lower in FactExtractor._PRONOUNS:
            return last_entity
        # Handle multi-word phrases starting with pronoun: "he, after the war" → entity
        if lower.split()[0] in FactExtractor._PRONOUNS:
            return last_entity
        return subject

    @staticmethod
    def _extract_first_named_entity(text: str) -> str | None:
        """Extract the first capitalized proper name from a sentence.

        Used to track the "current entity" for pronoun resolution across sentences.
        Skips pronouns, sentence-initial words, and common title-case words.
        """
        # Simple approach: find first group of capitalized words
        match = re.search(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})'
            r'(?:\s+(?:is|was|are|were|has|had|will|would|can|could|may|might|shall|should|'
            r'born|worked|studied|died|discovered|created|developed|earned|received|'
            r'won|attended|graduated|known|served|founded|married|located|published))',
            text,
        )
        if match:
            candidate = match.group(1).strip()
            # Skip if it's a pronoun
            if candidate.lower() in FactExtractor._PRONOUNS:
                return None
            return candidate
        return None

    @staticmethod
    def _looks_like_entity(text: str) -> bool:
        """Heuristic: does this text look like a named entity?

        Returns True if the text contains at least one capitalized word
        longer than 2 characters.
        """
        words = text.split()
        for w in words:
            if w[0].isupper() and len(w) > 2:
                return True
        return False


class CoverageEvaluator:
    """Evaluates how many source-document facts are captured by the KG.

    Three scored match levels:
      - exact_match:   (S, P, O) all match after normalization
      - entity_pair:   S & O exist in KG and are connected by any edge
      - entity_mention: at minimum S exists in KG

    Plus one diagnostic (unscored):
      - partial_match: S or O is a substring of a KG node (or vice versa)
    """

    def __init__(self) -> None:
        pass

    # ── Public API ────────────────────────────────────────────

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

        # ── Build normalized KG lookups ───────────────────────
        kg_nodes_norm: dict[str, str] = {}      # norm_name → original_name
        kg_node_original_lower: dict[str, str] = {}  # original_lower → original
        for n in graph.nodes():
            norm = self._normalize_name(n)
            kg_nodes_norm[norm] = n
            kg_node_original_lower[n.lower()] = n

        kg_triple_keys: set[tuple[str, str, str]] = {
            (self._normalize_name(s), p.lower().strip(), self._normalize_name(o))
            for s, p, o, _ in kg_triples
        }

        # Adjacency: (norm_subj, norm_obj) → True if any edge exists
        kg_adjacency: set[tuple[str, str]] = set()
        for s, _p, o, _src in kg_triples:
            kg_adjacency.add((
                self._normalize_name(s),
                self._normalize_name(o),
            ))

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

            subj = self._normalize_name(subj_raw)
            pred = pred_raw.lower().strip()
            obj = self._normalize_name(obj_raw)

            # 1. Exact triple match (normalized)
            exact = (subj, pred, obj) in kg_triple_keys

            # 2. Entity pair match: both in KG + connected
            subj_in_kg = subj in kg_nodes_norm
            obj_in_kg = obj in kg_nodes_norm
            connected = (subj, obj) in kg_adjacency if subj_in_kg and obj_in_kg else False
            pair_match = subj_in_kg and obj_in_kg and connected

            # 3. Entity mention match: at minimum subject in KG
            mention_match = subj_in_kg

            # 4. Partial/substring match (diagnostic only)
            partial_match = False
            if not mention_match:
                partial_match = self._is_partial_match(
                    subj_raw, kg_node_original_lower,
                )

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
                "partial_match": partial_match,
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

    # ── Name normalization ────────────────────────────────────

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize an entity name for comparison.

        - Lowercase
        - Strip leading articles (the, a, an)
        - Strip trailing punctuation
        - Collapse whitespace
        """
        if not name:
            return ""
        n = name.lower().strip()
        n = _LEADING_STRIP.sub('', n)
        n = _TRAILING_STRIP.sub('', n)
        n = _WHITESPACE.sub(' ', n).strip()
        return n

    # ── Partial / substring matching ──────────────────────────

    @staticmethod
    def _is_partial_match(
        fact_entity: str,
        kg_nodes_lower: dict[str, str],
    ) -> bool:
        """Check if a fact entity is a substring of any KG node name (or vice versa).

        This is a diagnostic (unscored) signal — it identifies cases where
        the entity likely exists in the KG but with a slightly different name.
        """
        fact_lower = fact_entity.lower().strip()
        if len(fact_lower) < 4:
            return False

        for kg_name_lower in kg_nodes_lower:
            if len(kg_name_lower) < 4:
                continue
            # Substring in either direction
            if fact_lower in kg_name_lower or kg_name_lower in fact_lower:
                return True
        return False

    # ── Interpretation ────────────────────────────────────────

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
