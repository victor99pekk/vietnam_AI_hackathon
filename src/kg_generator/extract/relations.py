"""Relation extraction between entities."""

import logging
from typing import Any

from kg_generator.config import Language
from kg_generator.extract.entities import Entity

logger = logging.getLogger(__name__)

# Common English relation patterns (subject, predicate, object)
RELATION_PATTERNS = [
    # (head_label, dep_label, predicate_label)
    ("PERSON", "ORG", "works_at"),
    ("PERSON", "GPE", "lives_in"),
    ("PERSON", "PERSON", "knows"),
    ("ORG", "GPE", "located_in"),
    ("ORG", "ORG", "subsidiary_of"),
    ("PERSON", "PRODUCT", "created"),
    ("ORG", "PRODUCT", "produces"),
    ("PERSON", "EVENT", "participated_in"),
]


class RelationExtractor:
    """Extracts (subject, predicate, object) triples from text."""

    def __init__(
        self,
        language: Language = Language.ENGLISH,
        use_llm: bool = False,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        self.language = language
        self.use_llm = use_llm
        self.model_name = model_name

    def extract(self, text: str, entities: list[Entity]) -> list[tuple[str, str, str]]:
        """Extract relation triples from text given extracted entities."""
        if self.use_llm:
            return self._llm_extract(text, entities)
        return self._rule_based_extract(text, entities)

    def _rule_based_extract(
        self, text: str, entities: list[Entity]
    ) -> list[tuple[str, str, str]]:
        """Rule-based relation extraction using entity co-occurrence and heuristics."""
        triples: list[tuple[str, str, str]] = []
        entity_map = {e.name.lower(): e for e in entities}

        # Use entity co-occurrence within the same sentence
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

        for sentence in sentences:
            sent_lower = sentence.lower()
            # Find entities that appear in this sentence
            present = [e for e in entities if e.name.lower() in sent_lower]
            if len(present) < 2:
                continue

            # Try pattern matching
            for e1 in present:
                for e2 in present:
                    if e1.name == e2.name:
                        continue
                    predicate = self._infer_predicate(sentence, e1, e2, entity_map)
                    if predicate:
                        triples.append((e1.name, predicate, e2.name))

        logger.debug(f"RuleBasedRelationExtractor: found {len(triples)} triples")
        return triples

    def _infer_predicate(
        self,
        sentence: str,
        e1: Entity,
        e2: Entity,
        entity_map: dict[str, Entity],
    ) -> str | None:
        """Infer the predicate between two co-occurring entities."""
        for head_label, dep_label, predicate in RELATION_PATTERNS:
            if e1.label == head_label and e2.label == dep_label:
                return predicate
            if e1.label == dep_label and e2.label == head_label:
                return predicate

        # Generic fallback based on labels
        if e1.label == e2.label:
            return "related_to"
        return "associated_with"

    def _llm_extract(
        self, text: str, entities: list[Entity]
    ) -> list[tuple[str, str, str]]:
        """LLM-based relation extraction (requires openai)."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai not installed — falling back to rule-based extraction")
            return self._rule_based_extract(text, entities)

        entity_list = "\n".join(f"- {e.name} ({e.label})" for e in entities)

        prompt = (
            f"Given the following text and extracted entities, identify all relationships "
            f"between entities. Output as a JSON list of [subject, predicate, object] triples.\n\n"
            f"Text:\n{text[:2000]}\n\n"
            f"Entities:\n{entity_list}\n\n"
            f"Return ONLY a JSON array of arrays, e.g.: "
            f'[["Alice", "works_at", "Acme Corp"], ["Bob", "lives_in", "Hanoi"]]'
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content or "[]"

        import json
        try:
            raw_triples = json.loads(content)
            triples = [
                (str(t[0]), str(t[1]), str(t[2]))
                for t in raw_triples
                if len(t) == 3
            ]
            logger.debug(f"LLM extraction: found {len(triples)} triples")
            return triples
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse LLM relation output")
            return []
