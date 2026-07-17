"""Relation extraction between entities."""

import logging
import re
from typing import Any

from kg_generator.config import Language
from kg_generator.extract.entities import Entity

logger = logging.getLogger(__name__)

# Common English relation patterns (subject_label, object_label, predicate)
RELATION_PATTERNS = [
    # Person ↔ Org
    ("PERSON", "ORG", "works_at"),
    ("PERSON", "ORG", "founded"),
    ("PERSON", "ORG", "studied_at"),
    # Person ↔ Location
    ("PERSON", "GPE", "lives_in"),
    ("PERSON", "GPE", "born_in"),
    ("PERSON", "GPE", "died_in"),
    # Person ↔ Person
    ("PERSON", "PERSON", "knows"),
    ("PERSON", "PERSON", "collaborated_with"),
    # Person ↔ Other
    ("PERSON", "PRODUCT", "created"),
    ("PERSON", "EVENT", "participated_in"),
    ("PERSON", "WORK_OF_ART", "authored"),
    ("PERSON", "DATE", "born_on"),
    # Org ↔ Location
    ("ORG", "GPE", "located_in"),
    ("ORG", "GPE", "headquartered_in"),
    # Org ↔ Org
    ("ORG", "ORG", "subsidiary_of"),
    ("ORG", "ORG", "partnered_with"),
    # Org ↔ Other
    ("ORG", "PRODUCT", "produces"),
    ("ORG", "EVENT", "organized"),
    ("ORG", "WORK_OF_ART", "published"),
    # Event ↔ Location / Date
    ("EVENT", "GPE", "took_place_in"),
    ("EVENT", "DATE", "occurred_on"),
    ("EVENT", "ORG", "involved"),
    # Work ↔ Person / Date
    ("WORK_OF_ART", "PERSON", "authored_by"),
    ("WORK_OF_ART", "DATE", "published_on"),
    # Concept ↔ Any
    ("CONCEPT", "PERSON", "associated_with"),
    ("CONCEPT", "ORG", "related_to"),
    ("CONCEPT", "GPE", "related_to"),
    ("CONCEPT", "CONCEPT", "related_to"),
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

    def extract(
        self,
        text: str,
        entities: list[Entity],
        source_chunk_id: str = "",
    ) -> list[tuple[str, str, str, str, str]]:
        """Extract relation triples from text given extracted entities.
        
        Returns: (subject_id, predicate, object_id, evidence_sentence, source_chunk_id).
        """
        if self.use_llm:
            return self._llm_extract(text, entities, source_chunk_id)
        return self._rule_based_extract(text, entities, source_chunk_id)

    def _rule_based_extract(
        self, text: str, entities: list[Entity], source_chunk_id: str = ""
    ) -> list[tuple[str, str, str, str, str]]:
        """Rule-based relation extraction using entity co-occurrence and heuristics."""
        triples: list[tuple[str, str, str, str, str]] = []
        entity_map = {e.name.lower(): e for e in entities}

        # Use entity co-occurrence within the same sentence
        sentences = self._sentences(text)

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
                        triples.append((e1.id, predicate, e2.id, sentence, source_chunk_id))

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
        self, text: str, entities: list[Entity], source_chunk_id: str = ""
    ) -> list[tuple[str, str, str, str, str]]:
        """LLM-based relation extraction using DeepSeek (OpenAI-compatible API)."""
        import os

        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai not installed — falling back to rule-based extraction")
            return self._rule_based_extract(text, entities, source_chunk_id)

        entity_list = "\n".join(f"- {e.name} ({e.label})" for e in entities)

        prompt = (
            f"Given the following text and extracted entities, identify all relationships "
            f"between entities. For each relationship, include the exact sentence from the "
            f"text that supports it. Output a JSON list of objects with subject, predicate, "
            f"object, and evidence fields.\n\n"
            f"Text:\n{text[:2000]}\n\n"
            f"Entities:\n{entity_list}\n\n"
            f"Return ONLY JSON, e.g. "
            f'[{"{"}"subject":"Alice","predicate":"works_at",'
            f'"object":"Acme Corp","evidence":"Alice works at Acme Corp."{"}"}]'
        )

        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content or "[]"

        import json
        try:
            raw_triples = json.loads(content)
            entity_ids = {entity.name.casefold(): entity.id for entity in entities}
            triples = []
            for triple in raw_triples:
                if isinstance(triple, dict):
                    subject_name = str(triple.get("subject", ""))
                    predicate = str(triple.get("predicate", ""))
                    object_name = str(triple.get("object", ""))
                    evidence = str(triple.get("evidence", "")).strip()
                elif isinstance(triple, list) and len(triple) >= 3:
                    subject_name, predicate, object_name = map(str, triple[:3])
                    evidence = str(triple[3]).strip() if len(triple) > 3 else ""
                else:
                    continue
                subject_id = entity_ids.get(subject_name.casefold())
                object_id = entity_ids.get(object_name.casefold())
                if subject_id and object_id:
                    if not evidence or evidence not in text:
                        evidence = self._find_evidence(text, subject_name, object_name)
                    triples.append(
                        (subject_id, predicate, object_id, evidence, source_chunk_id)
                    )
            logger.debug(f"LLM extraction: found {len(triples)} triples")
            return triples
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse LLM relation output")
            return []

    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Split text into evidence-sized sentences while retaining punctuation."""
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text) if part.strip()]

    @classmethod
    def _find_evidence(cls, text: str, subject: str, object_: str) -> str:
        """Find the sentence containing both relation endpoints."""
        subject_key = subject.casefold()
        object_key = object_.casefold()
        for sentence in cls._sentences(text):
            sentence_key = sentence.casefold()
            if subject_key in sentence_key and object_key in sentence_key:
                return sentence
        return ""
