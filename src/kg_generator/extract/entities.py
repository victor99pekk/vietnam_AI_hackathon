"""Entity extraction with language-swappable backends."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from kg_generator.config import Language
from kg_generator.identity import entity_id

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A single extracted entity with GraphRAG-ready properties.

    Aligned with best practices:
    - Identification:   id, name, aliases, type
    - Semantic:          description
    - Search:            embedding, importanceScore
    - Provenance:        source, confidenceScore, updatedAt
    """

    name: str
    label: str  # e.g., PERSON, ORG, LOCATION, CONCEPT, EVENT
    mentions: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    description: str = ""
    source: str = ""
    embedding: list[float] | None = None
    node_id: str = ""

    @property
    def id(self) -> str:
        """Stable Unicode-safe ID derived from entity type and canonical name."""
        return self.node_id or entity_id(self.label, self.name)

    @property
    def aliases(self) -> list[str]:
        """All known surface forms (synonyms, misspellings, abbreviations)."""
        return sorted(set(m.casefold() for m in self.mentions))

    @property
    def displayName(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Clean GraphRAG-ready dict with no legacy keys."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.label,
            "aliases": list(self.aliases),
            "description": self.description,
            "confidenceScore": self.confidence,
            "importanceScore": 0.0,  # populated by graph builder
            "source": [self.source] if self.source else [],
            "embedding": self.embedding,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }


class EntityExtractor(ABC):
    """Abstract base for entity extraction backends."""

    @abstractmethod
    def extract(self, text: str) -> list[Entity]:
        ...


class EnglishExtractor(EntityExtractor):
    """spaCy-based entity extraction for English."""

    def __init__(self, model_name: str = "en_core_web_lg") -> None:
        self.model_name = model_name
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self.model_name)
            except OSError:
                logger.warning(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Install with: python -m spacy download {self.model_name}"
                )
                import spacy
                self._nlp = spacy.blank("en")
        return self._nlp

    def extract(self, text: str) -> list[Entity]:
        doc = self.nlp(text)
        entities: list[Entity] = []
        seen = set()

        for ent in doc.ents:
            name = ent.text.strip()
            if name.lower() not in seen and len(name) > 1:
                seen.add(name.lower())
                entities.append(
                    Entity(
                        name=name,
                        label=ent.label_,
                        mentions=[name],
                        confidence=0.90,
                    )
                )

        # Also extract noun chunks as potential CONCEPT entities
        try:
            for chunk in doc.noun_chunks:
                name = chunk.text.strip()
                if name.lower() not in seen and len(name.split()) >= 1 and len(name) > 2:
                    seen.add(name.lower())
                    entities.append(
                        Entity(
                            name=name,
                            label="CONCEPT",
                            mentions=[name],
                            confidence=0.70,
                        )
                    )
        except ValueError:
            logger.debug("Noun chunks not available — skipping CONCEPT extraction")

        logger.debug(f"EnglishExtractor: found {len(entities)} entities")
        return entities


class VietnameseExtractor(EntityExtractor):
    """Underthesea-backed Vietnamese named-entity and noun-phrase extraction."""

    LABEL_MAP = {
        "PER": "PERSON",
        "LOC": "GPE",
        "ORG": "ORG",
    }

    def __init__(
        self,
        ner_function: Callable[[str], Sequence[Sequence[str]]] | None = None,
    ) -> None:
        if ner_function is None:
            try:
                from underthesea import ner
            except ImportError as error:
                raise RuntimeError(
                    "Vietnamese offline extraction requires Underthesea. "
                    "Install it with: uv sync --extra vi"
                ) from error
            ner_function = ner
        self._ner = ner_function

    def extract(self, text: str) -> list[Entity]:
        rows = [tuple(str(value) for value in row) for row in self._ner(text)]
        entities: list[Entity] = []
        seen: set[str] = set()

        for name, raw_label in self._tagged_spans(rows, tag_index=3):
            key = name.casefold()
            if key in seen or len(name) <= 1:
                continue
            seen.add(key)
            entities.append(
                Entity(
                    name=name,
                    label=self.LABEL_MAP.get(raw_label, raw_label),
                    mentions=[name],
                    confidence=0.85,
                )
            )

        for name, phrase_type in self._tagged_spans(rows, tag_index=2):
            if phrase_type != "NP":
                continue
            key = name.casefold()
            if key in seen or len(name) <= 2 or not any(char.isalpha() for char in name):
                continue
            seen.add(key)
            entities.append(
                Entity(
                    name=name,
                    label="CONCEPT",
                    mentions=[name],
                    confidence=0.70,
                )
            )

        logger.debug("VietnameseExtractor: found %d entities", len(entities))
        return entities

    @staticmethod
    def _tagged_spans(
        rows: Sequence[Sequence[str]],
        *,
        tag_index: int,
    ) -> list[tuple[str, str]]:
        """Reconstruct BIO spans from Underthesea token rows.

        A malformed I-tag starts a new span so one bad tag cannot discard the
        remaining sentence.
        """
        spans: list[tuple[str, str]] = []
        words: list[str] = []
        current_type = ""

        def flush() -> None:
            nonlocal words, current_type
            if words and current_type:
                spans.append((" ".join(words).strip(), current_type))
            words = []
            current_type = ""

        for row in rows:
            if len(row) <= tag_index:
                flush()
                continue
            word = row[0].strip()
            tag = row[tag_index].strip().upper()
            if not word or tag == "O" or "-" not in tag:
                flush()
                continue

            prefix, span_type = tag.split("-", 1)
            if prefix == "B" or prefix == "I" and span_type != current_type:
                flush()
                words = [word]
                current_type = span_type
            elif prefix == "I" and current_type == span_type:
                words.append(word)
            else:
                flush()

        flush()
        return spans


class SimpleExtractor(EntityExtractor):
    """Regex-based fallback extractor (no NLP deps required). Captures capitalized phrases."""

    # Common entity patterns
    CAPITALIZED_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
    EMAIL = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
    URL = re.compile(r"https?://[^\s]+")

    def extract(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        seen: set[str] = set()

        # Emails
        for m in self.EMAIL.finditer(text):
            name = m.group()
            if name.lower() not in seen:
                seen.add(name.lower())
                entities.append(Entity(name=name, label="CONTACT", mentions=[name]))

        # URLs
        for m in self.URL.finditer(text):
            name = m.group()
            if name.lower() not in seen:
                seen.add(name.lower())
                entities.append(Entity(name=name, label="URL", mentions=[name]))

        # Capitalized phrases (potential named entities)
        for m in self.CAPITALIZED_PHRASE.finditer(text):
            name = m.group()
            if name.lower() not in seen:
                seen.add(name.lower())
                entities.append(Entity(name=name, label="NAMED_ENTITY", mentions=[name], confidence=0.50))

        logger.debug(f"SimpleExtractor: found {len(entities)} entities")
        return entities


def get_extractor(language: Language, spacy_model: str = "en_core_web_sm") -> EntityExtractor:
    """Factory: return the appropriate extractor for the given language."""
    if language == Language.VIETNAMESE:
        return VietnameseExtractor()
    try:
        return EnglishExtractor(model_name=spacy_model)
    except ImportError:
        logger.warning("spaCy not available — falling back to SimpleExtractor")
        return SimpleExtractor()
