"""
Dataset generation for LLM fine-tuning evaluation.

Generates two parallel datasets from the same underlying information:
- **KG-Managed (Model B)**: Multi-hop QA pairs traversing the knowledge graph
- **Unmanaged (Model C)**: Flat QA pairs from raw source documents without KG structure

Key design principle: both datasets cover the same facts, but differ in structure.
This isolates "KG structure" as the single independent variable.
"""

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
        ("{rel} của {subj} là gì?", "{subj} {rel} {obj}."),
        ("{subj} {rel} ai/cái gì?", "{obj}."),
        ("{subj} có {rel} là gì?", "{subj} {rel} {obj}."),
    ],
}

_KG_MULTI_HOP_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": (
        "Given that {chain}, what {last_rel} {mid_node}?",
        "{answer}",
    ),
    "vi": (
        "Biết rằng {chain}, vậy {last_rel} của {mid_node} là gì?",
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
    ) -> None:
        self.language = language
        self.seed = seed
        self.max_hops = max_hops
        self.test_split = test_split
        random.seed(seed)

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
        triples: list[tuple[str, str, str, str]],
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Generate KG-structured QA pairs (Model B dataset).

        Returns paths to (train_file, test_file).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        qa_pairs: list[dict[str, Any]] = []
        qa_pairs.extend(self._single_hop_qa(graph, triples))
        qa_pairs.extend(self._multi_hop_qa(graph))
        qa_pairs.extend(self._comparison_qa(graph, entities))
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

        random.shuffle(unique)
        split_idx = int(len(unique) * (1 - self.test_split))
        train = unique[:split_idx]
        test = unique[split_idx:]

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
            target_count: if set, cap the dataset to this many pairs
                          (for token-volume matching with Model B)

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

        random.shuffle(unique)

        # Token-volume control: match KG dataset size if requested
        if target_count and len(unique) > target_count:
            unique = unique[:target_count]

        split_idx = int(len(unique) * (1 - self.test_split))
        train = unique[:split_idx]
        test = unique[split_idx:]

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
        self, graph: nx.DiGraph, triples: list[tuple[str, str, str, str]]
    ) -> list[dict[str, Any]]:
        """Generate single-hop factual questions from each triple."""
        qa_pairs: list[dict[str, Any]] = []

        for subj, pred, obj, source_text in triples:
            subj_label = self._node_label(graph, subj)
            obj_label = self._node_label(graph, obj)
            pred_readable = pred.replace("_", " ")

            for q_tmpl, a_tmpl in self._single_hop_tmpl:
                question = q_tmpl.format(subj=subj_label, rel=pred_readable, obj=obj_label)
                answer = a_tmpl.format(subj=subj_label, rel=pred_readable, obj=obj_label)
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "type": "single_hop",
                    "hops": 1,
                    "source": "kg",
                    "evidence": source_text,
                })

        return qa_pairs

    def _multi_hop_qa(self, graph: nx.DiGraph) -> list[dict[str, Any]]:
        """Generate 2-3 hop reasoning questions by traversing graph paths."""
        qa_pairs: list[dict[str, Any]] = []
        q_tmpl, a_tmpl = self._multi_hop_tmpl

        for start_node in list(graph.nodes())[:200]:  # Limit traversal
            paths = self._find_paths(graph, start_node, max_hops=self.max_hops)

            for path_nodes, path_edges in paths:
                if len(path_nodes) < 3:
                    continue  # Need at least 2 edges for multi-hop

                start = self._node_label(graph, path_nodes[0])
                end = self._node_label(graph, path_nodes[-1])

                # Build the chain description
                steps = []
                for i, (s, o) in enumerate(zip(path_nodes[:-1], path_nodes[1:])):
                    edge_data = graph.edges.get((s, o), {})
                    preds = edge_data.get("predicates", ["related_to"])
                    pred_readable = preds[0].replace("_", " ")
                    s_label = self._node_label(graph, s)
                    o_label = self._node_label(graph, o)
                    steps.append(f"{s_label} {pred_readable} {o_label}")

                # Question: given start + chain, ask for end
                if len(steps) >= 2:
                    chain_desc = ", ".join(steps[:-1])
                    last_step = steps[-1]
                    last_rel = last_step.split(" ", 1)[1] if " " in last_step else last_step
                    mid_node = self._node_label(graph, path_nodes[-2])

                    question = q_tmpl.format(chain=chain_desc, last_rel=last_rel, mid_node=mid_node, answer=end)
                    answer = a_tmpl.format(answer=end)

                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "type": "multi_hop",
                        "hops": len(steps),
                        "path": " → ".join(steps),
                        "source": "kg",
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

                    # Find shared relations
                    e1_neighbors = set(graph.successors(e1["name"])) | set(graph.predecessors(e1["name"]))
                    e2_neighbors = set(graph.successors(e2["name"])) | set(graph.predecessors(e2["name"]))

                    if e1_neighbors or e2_neighbors:
                        answer = self._describe_entity(graph, e1["name"])
                        qa_pairs.append({
                            "question": q_tmpl.format(e1=e1["name"], e2=e2["name"], answer=answer),
                            "answer": a_tmpl.format(answer=answer),
                            "type": "comparison",
                            "hops": 1,
                            "source": "kg",
                            "compare_with": e2["name"],
                        })

        return qa_pairs

    def _true_false_qa(
        self, graph: nx.DiGraph, triples: list[tuple[str, str, str, str]]
    ) -> list[dict[str, Any]]:
        """Generate true/false statements from KG facts + negative sampling."""
        qa_pairs: list[dict[str, Any]] = []
        q_tmpl, a_tmpl = self._true_false_tmpl

        # True statements from real triples
        for subj, pred, obj, _ in triples[:200]:
            subj_label = self._node_label(graph, subj)
            obj_label = self._node_label(graph, obj)
            pred_readable = pred.replace("_", " ")

            qa_pairs.append({
                "question": q_tmpl.format(subj=subj_label, rel=pred_readable, obj=obj_label, answer="True"),
                "answer": a_tmpl.format(answer="True"),
                "type": "true_false",
                "hops": 1,
                "source": "kg",
            })

        # False statements: corrupt the object
        all_nodes = list(graph.nodes())
        if len(all_nodes) > 2:
            for subj, pred, obj, _ in triples[:100]:
                # Pick a random wrong object
                wrong_obj = random.choice([n for n in all_nodes if n != obj and n != subj])
                subj_label = self._node_label(graph, subj)
                wrong_label = self._node_label(graph, wrong_obj)
                pred_readable = pred.replace("_", " ")

                qa_pairs.append({
                    "question": q_tmpl.format(subj=subj_label, rel=pred_readable, obj=wrong_label, answer="False"),
                    "answer": a_tmpl.format(answer="False"),
                    "type": "true_false",
                    "hops": 1,
                    "source": "kg",
                })

        return qa_pairs

    # ── Raw Text QA Generation ─────────────────────────────────

    def _extract_qa_from_text(
        self, text: str, source: str
    ) -> list[dict[str, Any]]:
        """Generate flat QA pairs from raw text using language-aware heuristics.

        Strategy: extract <subject, predicate, object> patterns from
        each sentence using language-specific regex, then template them into QA pairs.
        This produces simpler, single-hop questions compared to the KG version.
        """
        qa_pairs: list[dict[str, Any]] = []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            min_len = 15 if self.language == "vi" else 20  # Vietnamese sentences can be shorter
            if len(sent) < min_len:
                continue

            # ── Named entity detection ─────────────────────────
            # For English: capitalized words. For Vietnamese: any capitalized words
            # (Vietnamese proper nouns are typically capitalized, though not all nouns are)
            if self.language == "vi":
                # Vietnamese: match capitalized words (including those with diacritics)
                proper_nouns = re.findall(
                    r'\b([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]'
                    r'[a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+'
                    r'(?:\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]'
                    r'[a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+)*)\b',
                    sent,
                )
            else:
                proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', sent)

            # ── Copula/definition pattern (e.g. "X is Y" / "X là Y") ──
            is_match = re.match(self._copula_pattern, sent, re.IGNORECASE)
            if is_match and len(proper_nouns) >= 1:
                subject = is_match.group(1).strip()
                # For English: group 4 (after "a/an/the"). For Vietnamese: group 2 (after "là")
                if self.language == "vi":
                    predicate = is_match.group(2).strip().rstrip(".")
                else:
                    predicate = is_match.group(4).strip().rstrip(".")
                q_tmpl, a_tmpl = self._raw_def_tmpl
                qa_pairs.append({
                    "question": q_tmpl.format(subj=subject, pred=predicate),
                    "answer": a_tmpl.format(subj=subject, pred=predicate),
                    "type": "definition",
                    "hops": 1,
                    "source": "raw_text",
                    "evidence": sent,
                })

            # ── Proper noun pair → relationship question ────────
            if len(proper_nouns) >= 2:
                q_tmpl, a_tmpl = self._raw_rel_tmpl
                for i, pn1 in enumerate(proper_nouns[:3]):
                    for pn2 in proper_nouns[i+1:][:3]:
                        if pn1 != pn2:
                            qa_pairs.append({
                                "question": q_tmpl.format(e1=pn1, e2=pn2, sent=sent),
                                "answer": a_tmpl.format(sent=sent),
                                "type": "relationship",
                                "hops": 1,
                                "source": "raw_text",
                                "evidence": sent,
                            })

            # ── Factual patterns (e.g. "born in", "sinh tại") ──
            for pattern, q_template in self._raw_fact_patterns:
                match = re.search(pattern, sent, re.IGNORECASE)
                if match:
                    subject = match.group(1).strip()
                    fact = match.group(2).strip().rstrip(".")
                    qa_pairs.append({
                        "question": q_template.format(subject),
                        "answer": fact,
                        "type": "factual",
                        "hops": 1,
                        "source": "raw_text",
                        "evidence": sent,
                    })

        return qa_pairs

    # ── Helpers ─────────────────────────────────────────────────

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
        self, graph: nx.DiGraph, start: str, max_hops: int = 3
    ) -> list[tuple[list[str], list[tuple[str, str]]]]:
        """Find multi-hop paths from a start node up to max_hops."""
        paths: list[tuple[list[str], list[tuple[str, str]]]] = []

        def dfs(current: str, visited: list[str], edges: list[tuple[str, str]], depth: int):
            if depth > max_hops:
                return
            neighbors = list(graph.successors(current))
            for neighbor in neighbors:
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
    def _write_jsonl(data: list[dict[str, Any]], path: Path) -> Path:
        """Write a list of dicts to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path


def load_kg(path: Path) -> tuple[nx.DiGraph, list[dict[str, Any]], list[tuple[str, str, str, str]]]:
    """Load a knowledge graph from a JSON export file."""
    with open(path) as f:
        data = json.load(f)

    entities = data.get("entities", [])
    triples_raw = data.get("triples", [])
    graph = nx.node_link_graph(data.get("graph", {}), edges="edges")

    # Normalize triples to (subj, pred, obj, source_text) format
    triples: list[tuple[str, str, str, str]] = []
    for t in triples_raw:
        if isinstance(t, (list, tuple)):
            if len(t) >= 4:
                triples.append((str(t[0]), str(t[1]), str(t[2]), str(t[3])))
            elif len(t) == 3:
                triples.append((str(t[0]), str(t[1]), str(t[2]), ""))

    return graph, entities, triples


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
