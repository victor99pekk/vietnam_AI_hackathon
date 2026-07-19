"""
Dataset generation for LLM fine-tuning evaluation.

Generates two parallel datasets from the same underlying information:
- **KG-Managed (Model B)**: Multi-hop QA pairs traversing the knowledge graph
- **Unmanaged (Model C)**: Flat QA pairs from raw source documents without KG structure

Key design principle: both datasets use the same cleaned source chunks, the
same source-level train/test split, and matched estimated training-token
budgets. Their QA transformations differ: graph templates for Model B and
source-text-only heuristics for Model C.
"""

import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# ── Language-specific QA templates ────────────────────────────
# Each template set is a list of (question_template, answer_template) strings.
# Placeholders: {subj}, {rel}, {obj}, {chain}, {last_rel}, {mid_node}

_KG_SINGLE_HOP_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("What is the {rel} of {subj}?", "{subj} {rel} {obj}."),
        ("{subj} {rel} what?", "{obj}."),
        ("Who or what does {subj} {rel}?", "{subj} {rel} {obj}."),
    ],
    "vi": [
        ("{subj} và {obj} có mối quan hệ gì?", "{desc}"),
        ("Mô tả mối liên hệ giữa {subj} và {obj}.", "{desc}"),
        ("Theo văn bản, {subj} liên quan đến {obj} như thế nào?", "{desc}"),
    ],
}

_KG_MULTI_HOP_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": (
        "Starting from {start}, follow these relationships in order: {relations}. Which entity is reached?",
        "{answer}",
    ),
    "vi": (
        "Bắt đầu từ {start}, lần lượt đi theo các quan hệ sau: {relations}. Ta đến thực thể nào?",
        "{answer}",
    ),
}

_KG_COMPARISON_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": (
        "Compare {e1} and {e2}. What do they have in common?",
        "{answer}",
    ),
    "vi": (
        "So sánh {e1} và {e2}. Chúng có điểm gì chung?",
        "{answer}",
    ),
}

_KG_TRUE_FALSE_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": (
        "True or False: {subj} {rel} {obj}.",
        "{answer}",
    ),
    "vi": (
        "Đúng hay Sai: {subj} {rel} {obj}.",
        "{answer}",
    ),
}

_RAW_DEFINITION_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": ("What is {subj}?", "{pred}"),
    "vi": ("{subj} là gì?", "{pred}"),
}

_RAW_RELATIONSHIP_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": ("What is the relationship between {e1} and {e2}?", "{sent}"),
    "vi": ("Mối quan hệ giữa {e1} và {e2} là gì?", "{sent}"),
}

# ── Language-specific regex patterns for raw-text extraction ───
# Each entry: (pattern, question_template)
#   pattern: regex with two capture groups (subject, object)
#   question_template: string with {0}=subject placeholder

_RAW_FACT_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "en": [
        (r'(.+?)\s+(?:was\s+)?born\s+in\s+(.+?)(?:,|\.|$)', "Where was {0} born?"),
        (r'(.+?)\s+(?:worked|works)\s+(?:at|for|in)\s+(.+?)(?:,|\.|$)', "Where did {0} work?"),
        (r'(.+?)\s+(?:studied|studies)\s+(?:at|in)\s+(.+?)(?:,|\.|$)', "Where did {0} study?"),
        (r'(.+?)\s+(?:died|dies)\s+in\s+(.+?)(?:,|\.|$)', "Where did {0} die?"),
        (r'(.+?)\s+(?:discovered|invented|created|developed)\s+(.+?)(?:,|\.|$)', "What did {0} discover?"),
    ],
    "vi": [
        (r'(.+?)\s+sinh\s+(?:ra\s+)?(?:tại|ở)\s+(.+?)(?:,|\.|$)', "{0} sinh ra ở đâu?"),
        (r'(.+?)\s+(?:làm việc|làm)\s+(?:tại|ở|cho)\s+(.+?)(?:,|\.|$)', "{0} làm việc ở đâu?"),
        (r'(.+?)\s+(?:học|học tập)\s+(?:tại|ở)\s+(.+?)(?:,|\.|$)', "{0} học ở đâu?"),
        (r'(.+?)\s+(?:mất|qua đời)\s+(?:tại|ở)\s+(.+?)(?:,|\.|$)', "{0} mất ở đâu?"),
        (r'(.+?)\s+(?:phát hiện|phát minh|sáng tạo|tạo ra)\s+(.+?)(?:,|\.|$)', "{0} đã phát hiện/phát minh ra gì?"),
        (r'(.+?)\s+là\s+(?:một\s+)?(.+?)(?:,|\.|$)', "{0} là gì?"),
    ],
}

# ── Language-specific copula / definition patterns ─────────────
# pattern with two capture groups: (subject, rest_of_sentence)

_COPULA_PATTERNS: dict[str, str] = {
    "en": r'(.+?)\s+(is|was|are|were)\s+(a|an|the)?\s*(.+)',
    "vi": r'(.+?)\s+là\s+(?:một\s+)?(.+)',
}

_PRONOUNS: dict[str, set[str]] = {
    "en": {"he", "she", "his", "her", "they", "their", "it", "its", "this", "that"},
    "vi": {"anh", "chị", "ông", "bà", "họ", "nó", "cô", "người", "tổ chức này", "đây"},
}

_STRUCTURAL_PREDICATES = {"NEXT", "PART_OF", "MENTIONS"}


def estimate_qa_tokens(item: dict[str, Any]) -> int:
    """Return a deterministic tokenizer-independent QA token estimate."""
    question = str(item.get("question", item.get("instruction", "")))
    answer = str(item.get("answer", item.get("response", "")))
    return max(1, len(re.findall(r"\w+|[^\w\s]", f"{question}\n{answer}", re.UNICODE)))


def balance_jsonl_token_volume(
    first_path: Path, second_path: Path
) -> dict[str, dict[str, int]]:
    """Trim the larger training dataset so both have comparable token volume.

    Input order is already deterministically shuffled by QADatasetGenerator.
    Files are rewritten only inside the current experiment output directory.
    """

    def read(path: Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def token_total(items: list[dict[str, Any]]) -> int:
        return sum(estimate_qa_tokens(item) for item in items)

    def trim(items: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        used = 0
        for item in items:
            item_tokens = estimate_qa_tokens(item)
            if selected and used + item_tokens > budget:
                continue
            selected.append(item)
            used += item_tokens
            if used >= budget:
                break
        return selected

    first = read(first_path)
    second = read(second_path)
    if not first or not second:
        raise ValueError("Both KG and raw-text training datasets must be non-empty")

    target = min(token_total(first), token_total(second))
    first = trim(first, target)
    second = trim(second, target)
    QADatasetGenerator._write_jsonl(first, first_path)
    QADatasetGenerator._write_jsonl(second, second_path)

    return {
        "kg": {"examples": len(first), "estimated_tokens": token_total(first)},
        "raw": {"examples": len(second), "estimated_tokens": token_total(second)},
        "target_estimated_tokens": {"value": target},
    }


class QADatasetGenerator:
    """Generates QA pairs from a knowledge graph and its source documents.

    Uses template-based generation (no external LLM needed) to keep the
    pipeline lightweight and deterministic. Supports English and Vietnamese
    templates for both KG-structured and raw-text QA generation.

    Parameters:
        language: "en" or "vi" — selects language-specific templates and regex.
        seed: Random seed for reproducibility.
        max_hops: Maximum hop depth for multi-hop KG questions.
        test_split: Fraction of data reserved for test set.
    """

    def __init__(
        self,
        language: str = "en",
        seed: int = 42,
        max_hops: int = 3,
        test_split: float = 0.2,
        max_triples: int = 2000,
        max_paths_per_start: int = 20,
        max_pairs: int = 10000,
    ) -> None:
        self.language = language
        self.seed = seed
        self.max_hops = max_hops
        self.test_split = test_split
        self.max_triples = max_triples
        self.max_paths_per_start = max_paths_per_start
        self.max_pairs = max_pairs
        self._rng = random.Random(seed)

        # Resolve template sets — fall back to English for unknown languages
        self._single_hop_tmpl = _KG_SINGLE_HOP_TEMPLATES.get(language, _KG_SINGLE_HOP_TEMPLATES["en"])
        self._multi_hop_tmpl = _KG_MULTI_HOP_TEMPLATES.get(language, _KG_MULTI_HOP_TEMPLATES["en"])
        self._comparison_tmpl = _KG_COMPARISON_TEMPLATES.get(language, _KG_COMPARISON_TEMPLATES["en"])
        self._true_false_tmpl = _KG_TRUE_FALSE_TEMPLATES.get(language, _KG_TRUE_FALSE_TEMPLATES["en"])
        self._raw_def_tmpl = _RAW_DEFINITION_TEMPLATES.get(language, _RAW_DEFINITION_TEMPLATES["en"])
        self._raw_rel_tmpl = _RAW_RELATIONSHIP_TEMPLATES.get(language, _RAW_RELATIONSHIP_TEMPLATES["en"])
        self._raw_fact_patterns = _RAW_FACT_PATTERNS.get(language, _RAW_FACT_PATTERNS["en"])
        self._copula_pattern = _COPULA_PATTERNS.get(language, _COPULA_PATTERNS["en"])

    # ── Public API ──────────────────────────────────────────────

    def generate_from_kg(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str, str]],
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Generate KG-structured QA pairs (Model B dataset).

        Returns paths to (train_file, test_file).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        qa_pairs: list[dict[str, Any]] = []
        qa_pairs.extend(self._single_hop_qa(graph, triples))
        qa_pairs.extend(self._multi_hop_qa(graph))
        qa_pairs.extend(self._true_false_qa(graph, triples))

        # Deduplicate by question
        seen = set()
        unique: list[dict[str, Any]] = []
        for qa in qa_pairs:
            key = qa["question"].strip().lower()
            if key not in seen:
                seen.add(key)
                if qa["answer"].strip():  # Skip empty answers
                    unique.append(qa)

        if len(unique) > self.max_pairs:
            self._rng.shuffle(unique)
            unique = unique[: self.max_pairs]

        train, test = self._grouped_split(unique)

        data_dir = output_dir / "test_training_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_path = self._write_jsonl(train, data_dir / "kg_qa_train.jsonl")
        test_path = self._write_jsonl(test, data_dir / "kg_qa_test.jsonl")

        logger.info(
            "KG-Managed dataset: %d train + %d test QA pairs → %s",
            len(train), len(test), output_dir,
        )
        return train_path, test_path

    def generate_from_raw_text(
        self,
        documents: list[dict[str, str]],
        output_dir: Path,
        target_count: int | None = None,
    ) -> tuple[Path, Path]:
        """Generate flat QA pairs from raw documents (Model C dataset).

        Parameters:
            documents: list of {"content": str, "source": str} dicts
            output_dir: where to write the output files
            target_count: backward-compatible optional example-count cap.
                          Method 2 now balances final training files by tokens.

        Returns paths to (train_file, test_file).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        qa_pairs: list[dict[str, Any]] = []
        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            qa_pairs.extend(self._extract_qa_from_text(content, source))

        # Deduplicate
        seen = set()
        unique: list[dict[str, Any]] = []
        for qa in qa_pairs:
            key = qa["question"].strip().lower()
            if key not in seen and qa["answer"].strip():
                seen.add(key)
                unique.append(qa)

        if len(unique) > self.max_pairs:
            self._rng.shuffle(unique)
            unique = unique[: self.max_pairs]

        self._rng.shuffle(unique)

        # Token-volume control: match KG dataset size if requested
        if target_count and len(unique) > target_count:
            unique = unique[:target_count]

        train, test = self._grouped_split(unique)

        data_dir = output_dir / "test_training_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_path = self._write_jsonl(train, data_dir / "raw_qa_train.jsonl")
        test_path = self._write_jsonl(test, data_dir / "raw_qa_test.jsonl")

        logger.info(
            "Unmanaged dataset: %d train + %d test QA pairs → %s",
            len(train), len(test), output_dir,
        )
        return train_path, test_path

    # ── KG QA Generation ───────────────────────────────────────

    def _single_hop_qa(
        self, graph: nx.DiGraph, triples: list[tuple[str, str, str, str, str]]
    ) -> list[dict[str, Any]]:
        """Generate single-hop factual questions from each triple."""
        qa_pairs: list[dict[str, Any]] = []

        selected_triples = list(triples)
        if len(selected_triples) > self.max_triples:
            selected_triples = self._rng.sample(selected_triples, self.max_triples)
        for subj, pred, obj, source_text, source_chunk_id in selected_triples:
            subj_label = self._node_label(graph, subj)
            obj_label = self._node_label(graph, obj)
            # Use the description field (source_text) as the relation meaning.
            # GraphGen stores all predicates as "RELATION" — the actual
            # semantic content is in the description.
            description = source_text.strip() if source_text else f"{subj_label} liên quan đến {obj_label}"
            pred_readable = pred.replace("_", " ")

            for q_tmpl, a_tmpl in self._single_hop_tmpl:
                question = q_tmpl.format(
                    subj=subj_label, rel=pred_readable, obj=obj_label, desc=description,
                )
                answer = a_tmpl.format(
                    subj=subj_label, rel=pred_readable, obj=obj_label, desc=description,
                )
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "type": "single_hop",
                    "hops": 1,
                    "source": "kg",
                    "evidence": source_text,
                    "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
                    "_group_id": (
                        f"source:{source_chunk_id}"
                        if source_chunk_id else f"triple:{subj}|{pred}|{obj}"
                    ),
                })

        return qa_pairs

    def _multi_hop_qa(self, graph: nx.DiGraph) -> list[dict[str, Any]]:
        """Generate 2-3 hop reasoning questions by traversing graph paths."""
        qa_pairs: list[dict[str, Any]] = []
        q_tmpl, a_tmpl = self._multi_hop_tmpl

        for start_node in list(graph.nodes())[:200]:  # Limit traversal
            paths = self._find_paths(
                graph,
                start_node,
                max_hops=self.max_hops,
                max_paths=self.max_paths_per_start,
            )

            for path_nodes, path_edges in paths:
                if len(path_nodes) < 3:
                    continue  # Need at least 2 edges for multi-hop

                start = self._node_label(graph, path_nodes[0])
                end = self._node_label(graph, path_nodes[-1])

                # Build the relation sequence without revealing intermediate nodes.
                steps = []
                relations = []
                source_chunk_ids: set[str] = set()
                for i, (s, o) in enumerate(zip(path_nodes[:-1], path_nodes[1:])):
                    edge_data = graph.edges.get((s, o), {})
                    predicates = self._domain_predicates(edge_data)
                    if not predicates:
                        steps = []
                        break
                    pred_readable = predicates[0].replace("_", " ")
                    s_label = self._node_label(graph, s)
                    o_label = self._node_label(graph, o)
                    steps.append(f"{s_label} {pred_readable} {o_label}")
                    relations.append(pred_readable)
                    source_chunk_ids.update(
                        str(chunk_id)
                        for chunk_id in edge_data.get("source_chunk_ids", [])
                        if chunk_id
                    )

                # Following only the named relation sequence requires graph traversal.
                if len(steps) >= 2:
                    source_groups = [f"source:{chunk_id}" for chunk_id in sorted(source_chunk_ids)]
                    assignments = {self._is_test_group(group) for group in source_groups}
                    if len(assignments) > 1:
                        continue
                    question = q_tmpl.format(
                        start=start,
                        relations=" → ".join(relations),
                        answer=end,
                    )
                    answer = a_tmpl.format(answer=end)

                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "type": "multi_hop",
                        "hops": len(steps),
                        "path": " → ".join(steps),
                        "source": "kg",
                        "source_chunk_ids": sorted(source_chunk_ids),
                        "_group_id": (
                            source_groups[0]
                            if source_groups else "path:" + "|".join(path_nodes)
                        ),
                    })

        return qa_pairs

    def _comparison_qa(
        self, graph: nx.DiGraph, entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate comparison questions between entities of the same type."""
        qa_pairs: list[dict[str, Any]] = []
        q_tmpl, a_tmpl = self._comparison_tmpl

        # Group entities by type
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in entities:
            etype = e.get("type", "ENTITY")
            if etype not in ("Chunk", "Document"):
                by_type[etype].append(e)

        for etype, ents in by_type.items():
            if len(ents) < 2:
                continue

            # Pair up entities of the same type
            for i in range(min(len(ents), 10)):
                for j in range(i + 1, min(len(ents), 10)):
                    e1, e2 = ents[i], ents[j]
                    e1_id = str(e1.get("id", e1.get("name", "")))
                    e2_id = str(e2.get("id", e2.get("name", "")))
                    if e1_id not in graph or e2_id not in graph:
                        continue
                    e1_label = self._node_label(graph, e1_id)
                    e2_label = self._node_label(graph, e2_id)

                    # Find shared relations
                    e1_neighbors = set(graph.successors(e1_id)) | set(graph.predecessors(e1_id))
                    e2_neighbors = set(graph.successors(e2_id)) | set(graph.predecessors(e2_id))

                    shared_neighbors = sorted(e1_neighbors & e2_neighbors)
                    if shared_neighbors:
                        answer = ", ".join(
                            self._node_label(graph, node_id)
                            for node_id in shared_neighbors[:5]
                        )
                        qa_pairs.append({
                            "question": q_tmpl.format(e1=e1_label, e2=e2_label, answer=answer),
                            "answer": a_tmpl.format(answer=answer),
                            "type": "comparison",
                            "hops": 1,
                            "source": "kg",
                            "compare_with": e2_label,
                            "_group_id": f"comparison:{e1_id}|{e2_id}",
                        })

        return qa_pairs

    def _true_false_qa(
        self, graph: nx.DiGraph, triples: list[tuple[str, str, str, str, str]]
    ) -> list[dict[str, Any]]:
        """Generate true/false statements from KG facts + negative sampling."""
        qa_pairs: list[dict[str, Any]] = []
        q_tmpl, a_tmpl = self._true_false_tmpl
        true_answer = "Đúng" if self.language == "vi" else "True"
        false_answer = "Sai" if self.language == "vi" else "False"
        known_triples = {(subj, pred, obj) for subj, pred, obj, _, _ in triples}

        # True statements from real triples
        for subj, pred, obj, source_text, source_chunk_id in triples[:200]:
            subj_label = self._node_label(graph, subj)
            obj_label = self._node_label(graph, obj)
            pred_readable = pred.replace("_", " ")

            qa_pairs.append({
                "question": q_tmpl.format(subj=subj_label, rel=pred_readable, obj=obj_label, answer=true_answer),
                "answer": a_tmpl.format(answer=true_answer),
                "type": "true_false",
                "hops": 1,
                "source": "kg",
                "evidence": source_text,
                "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
                "_group_id": (
                    f"source:{source_chunk_id}"
                    if source_chunk_id else f"triple:{subj}|{pred}|{obj}"
                ),
            })

        # False statements: corrupt the object
        all_nodes = list(graph.nodes())
        if len(all_nodes) > 2:
            for subj, pred, obj, source_text, source_chunk_id in triples[:100]:
                object_type = graph.nodes[obj].get("type") if obj in graph else None
                candidates = [
                    node for node in all_nodes
                    if node not in {obj, subj}
                    and (subj, pred, node) not in known_triples
                    and (
                        object_type is None
                        or graph.nodes[node].get("type") == object_type
                    )
                ]
                if not candidates:
                    candidates = [
                        node for node in all_nodes
                        if node not in {obj, subj}
                        and (subj, pred, node) not in known_triples
                    ]
                if not candidates:
                    continue
                wrong_obj = self._rng.choice(candidates)
                subj_label = self._node_label(graph, subj)
                wrong_label = self._node_label(graph, wrong_obj)
                pred_readable = pred.replace("_", " ")

                qa_pairs.append({
                    "question": q_tmpl.format(subj=subj_label, rel=pred_readable, obj=wrong_label, answer=false_answer),
                    "answer": a_tmpl.format(answer=false_answer),
                    "type": "true_false",
                    "hops": 1,
                    "source": "kg",
                    "evidence": source_text,
                    "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
                    "_group_id": (
                        f"source:{source_chunk_id}"
                        if source_chunk_id else f"triple:{subj}|{pred}|{obj}"
                    ),
                })

        return qa_pairs

    # ── Raw Text QA Generation ─────────────────────────────────

    def _extract_qa_from_text(
        self, text: str, source: str
    ) -> list[dict[str, Any]]:
        """Generate flat QA pairs from raw text using language-aware heuristics.

        Strategy: use source-grounded questions as the primary approach
        (\"what does the text say about X?\"), supplemented by copula-based
        definition questions where the pattern is clear.  This avoids the
        noisy proper-noun-pairing and regex-factual approaches that produce
        incoherent pairs for Vietnamese.
        """
        qa_pairs: list[dict[str, Any]] = []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            min_len = 15 if self.language == "vi" else 20
            if len(sent) < min_len or len(sent) > 600:
                continue

            group_id = f"source:{source}"

            # ── Detect candidate subjects ──────────────────────
            # For Vietnamese we rely on the copula pattern (X là Y) and
            # capitalized multi-word phrases as proper-noun proxies.
            candidates: list[str] = []

            # Copula pattern: extract the subject
            is_match = re.match(self._copula_pattern, sent, re.IGNORECASE)
            if is_match:
                subject = is_match.group(1).strip()
                if self.language == "vi":
                    predicate = is_match.group(2).strip().rstrip(".")
                else:
                    predicate = is_match.group(4).strip().rstrip(".")
                # Quality gate: subject must look like a real entity
                if len(subject.split()) >= 2 or len(subject) > 12:
                    candidates.append(subject)
                    # Also generate a definition-style pair
                    q_tmpl, a_tmpl = self._raw_def_tmpl
                    qa_pairs.append({
                        "question": q_tmpl.format(subj=subject, pred=predicate),
                        "answer": a_tmpl.format(subj=subject, pred=predicate),
                        "type": "definition",
                        "hops": 1,
                        "source": "raw_text",
                        "source_id": source,
                        "evidence": sent,
                        "_group_id": group_id,
                    })

            # Proper-noun heuristic (capitalized words)
            if self.language == "vi":
                proper_nouns = re.findall(
                    r'\b([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]'
                    r'[a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+'
                    r'(?:\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]'
                    r'[a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+)*)\b',
                    sent,
                )
            else:
                proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', sent)

            pronouns = _PRONOUNS.get(self.language, _PRONOUNS["en"])
            proper_nouns = [
                noun for noun in proper_nouns
                if noun.strip().lower() not in pronouns
            ]
            # Only keep multi-word or long proper nouns as candidates
            for noun in proper_nouns:
                if len(noun.split()) >= 2 or len(noun) > 12:
                    if noun not in candidates:
                        candidates.append(noun)

            # ── Source-grounded question (primary type) ─────────
            # For each candidate entity in the sentence, generate a
            # \"what does the source say about X?\" question.
            for subject in candidates[:2]:  # limit to 2 candidates per sentence
                question = (
                    f"Theo văn bản, thông tin nào được nêu về {subject}?"
                    if self.language == "vi"
                    else f"According to the source, what is stated about {subject}?"
                )
                qa_pairs.append({
                    "question": question,
                    "answer": sent,
                    "type": "source_grounded",
                    "hops": 1,
                    "source": "raw_text",
                    "source_id": source,
                    "evidence": sent,
                    "_group_id": group_id,
                })

        return qa_pairs

    # ── Helpers ─────────────────────────────────────────────────

    def _grouped_split(
        self, items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split by provenance group so paraphrases of one fact never leak.

        Uses two-phase assignment: first assign groups to test/train by hash,
        then rebalance if the resulting test fraction deviates too far from
        the configured test_split (guards against a few 'chatty' groups
        dominating one side).
        """
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for index, item in enumerate(items):
            grouped[str(item.get("_group_id", f"item:{index}"))].append(item)

        group_ids = sorted(grouped)
        if len(group_ids) <= 1:
            train_ids = set(group_ids)
            test_ids: set[str] = set()
        else:
            test_ids = {g for g in group_ids if self._is_test_group(g)}
            if not test_ids:
                test_ids = {max(group_ids, key=self._group_score)}
            if len(test_ids) == len(group_ids):
                test_ids.remove(min(group_ids, key=self._group_score))

            # ── Rebalance: swap groups between test/train if the split is too lopsided ──
            total_items = len(items)
            test_count = sum(len(grouped[g]) for g in test_ids)
            test_frac = test_count / total_items if total_items else 0
            tolerance = 0.15  # allow ±15% deviation from target split

            if test_frac > self.test_split + tolerance:
                # Too many items in test — move some test groups back to train
                overflow_groups = sorted(
                    test_ids, key=lambda g: len(grouped[g]), reverse=True,
                )
                for g in overflow_groups:
                    if len(test_ids) <= 1:
                        break
                    test_ids.discard(g)
                    test_count = sum(len(grouped[g2]) for g2 in test_ids)
                    if test_count / total_items <= self.test_split + tolerance:
                        break
            elif test_frac < max(0.05, self.test_split - tolerance):
                # Too few items in test — move some train groups to test
                train_only = [g for g in group_ids if g not in test_ids]
                need_groups = sorted(
                    train_only, key=lambda g: len(grouped[g]),
                )
                for g in need_groups:
                    test_ids.add(g)
                    test_count = sum(len(grouped[g2]) for g2 in test_ids)
                    if test_count / total_items >= self.test_split - tolerance:
                        break

            train_ids = set(group_ids) - test_ids

        train: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []
        for group_id in group_ids:
            target = train if group_id in train_ids else test
            for item in grouped[group_id]:
                clean = dict(item)
                clean.pop("_group_id", None)
                target.append(clean)
        self._rng.shuffle(train)
        self._rng.shuffle(test)

        actual_test_frac = len(test) / (len(train) + len(test)) if (len(train) + len(test)) else 0
        logger.info(
            "Grouped split: %d train + %d test (%.1f%% test, target %.0f%%)",
            len(train), len(test), actual_test_frac * 100, self.test_split * 100,
        )
        return train, test

    def _group_score(self, group_id: str) -> float:
        digest = hashlib.sha256(f"{self.seed}:{group_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    def _is_test_group(self, group_id: str) -> bool:
        return self._group_score(group_id) < self.test_split

    @staticmethod
    def _node_label(graph: nx.DiGraph, node_id: str) -> str:
        """Get a readable label for a graph node."""
        if node_id in graph.nodes():
            data = graph.nodes[node_id]
            return data.get("name", node_id)
        return node_id

    @staticmethod
    def _describe_entity(graph: nx.DiGraph, node_id: str) -> str:
        """Get a description for an entity node."""
        if node_id in graph.nodes():
            data = graph.nodes[node_id]
            desc = data.get("description", "")
            if desc:
                return desc[:200]
        return node_id

    def _find_paths(
        self,
        graph: nx.DiGraph,
        start: str,
        max_hops: int = 3,
        max_paths: int = 20,
    ) -> list[tuple[list[str], list[tuple[str, str]]]]:
        """Find multi-hop paths from a start node up to max_hops."""
        paths: list[tuple[list[str], list[tuple[str, str]]]] = []

        def dfs(current: str, visited: list[str], edges: list[tuple[str, str]], depth: int):
            if depth > max_hops or len(paths) >= max_paths:
                return
            neighbors = [
                neighbor for neighbor in graph.successors(current)
                if self._domain_predicates(graph.edges.get((current, neighbor), {}))
            ]
            for neighbor in neighbors:
                if len(paths) >= max_paths:
                    break
                if neighbor in visited:
                    continue
                new_visited = visited + [neighbor]
                new_edges = edges + [(current, neighbor)]
                if len(new_visited) >= 2:
                    paths.append((new_visited, new_edges))
                if depth < max_hops:
                    dfs(neighbor, new_visited, new_edges, depth + 1)

        dfs(start, [start], [], 1)
        return paths

    @staticmethod
    def _domain_predicates(edge_data: dict[str, Any]) -> list[str]:
        return [
            str(predicate)
            for predicate in edge_data.get("predicates", ["related_to"])
            if str(predicate).upper() not in _STRUCTURAL_PREDICATES
        ]

    @staticmethod
    def _write_jsonl(data: list[dict[str, Any]], path: Path) -> Path:
        """Write a list of dicts to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path


def load_kg(path: Path) -> tuple[
    nx.DiGraph,
    list[dict[str, Any]],
    list[tuple[str, str, str, str, str]],
]:
    """Load a knowledge graph from a JSON export file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities", [])
    triples_raw = data.get("triples", [])
    graph_data = data.get("graph", {})
    graph = nx.node_link_graph(graph_data, edges="edges")

    # Normalize to (subj, pred, obj, source_text, source_chunk_id).
    triples: list[tuple[str, str, str, str, str]] = []
    for t in triples_raw:
        if isinstance(t, dict):
            subj = str(t.get("subject", ""))
            pred = str(t.get("predicate", ""))
            obj = str(t.get("object", ""))
            source_text = str(
                t.get("evidence_sentence") or t.get("description") or ""
            )
            source_chunk_value = t.get("source_chunk_id") or t.get("source_chunk_ids") or ""
            if isinstance(source_chunk_value, (list, tuple, set)):
                source_chunk_value = next(iter(source_chunk_value), "")
            source_chunk_id = str(source_chunk_value)
            if subj and obj and pred.upper() not in _STRUCTURAL_PREDICATES:
                triples.append((subj, pred, obj, source_text, source_chunk_id))
        elif isinstance(t, (list, tuple)):
            if len(t) >= 4:
                if str(t[1]).upper() not in _STRUCTURAL_PREDICATES:
                    triples.append((
                        str(t[0]), str(t[1]), str(t[2]), str(t[3]),
                        str(t[4]) if len(t) > 4 else "",
                    ))
            elif len(t) == 3:
                if str(t[1]).upper() not in _STRUCTURAL_PREDICATES:
                    triples.append((str(t[0]), str(t[1]), str(t[2]), "", ""))

    return graph, entities, triples


def load_raw_documents_from_kg(path: Path) -> list[dict[str, str]]:
    """Recover cleaned source chunks embedded in a KG export.

    Model C receives these texts only. It does not consume entities, triples,
    relations, or graph paths, while still seeing the same processed corpus as
    Model B.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    documents: list[dict[str, str]] = []
    for node in data.get("graph", {}).get("nodes", []):
        if node.get("type") != "Chunk":
            continue
        content = str(node.get("text", "")).strip()
        if not content:
            continue
        documents.append({
            "content": content,
            "source": str(node.get("id", node.get("source", "unknown"))),
        })
    return documents


def load_raw_documents(paths: list[Path]) -> list[dict[str, str]]:
    """Load raw text documents from files/directories.

    Supports .txt, .jsonl (one JSON object per line, reads "text" or "content" field),
    and .json (array of objects or single object with "text"/"content").
    """
    docs: list[dict[str, str]] = []
    for path in paths:
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.suffix in (".txt", ".jsonl", ".json"):
                    docs.extend(_load_single_document_file(file_path))
        elif path.suffix in (".txt", ".jsonl", ".json"):
            docs.extend(_load_single_document_file(path))
    return docs


def _load_single_document_file(path: Path) -> list[dict[str, str]]:
    """Load documents from a single file, dispatching by extension."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return [{"content": path.read_text(encoding="utf-8"), "source": str(path)}]

    if suffix == ".jsonl":
        docs: list[dict[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = obj.get("text", obj.get("content", ""))
                if content:
                    docs.append({"content": content, "source": str(path)})
        return docs

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [
                {"content": item.get("text", item.get("content", "")), "source": str(path)}
                for item in data
                if item.get("text") or item.get("content")
            ]
        content = data.get("text", data.get("content", ""))
        if content:
            return [{"content": content, "source": str(path)}]
        return []

    return []
