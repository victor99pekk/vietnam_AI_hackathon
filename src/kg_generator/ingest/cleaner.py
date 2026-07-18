"""Text normalization and cleaning with language-specific backends."""

import re
import logging
import unicodedata
from abc import ABC, abstractmethod

from kg_generator.config import Language
from kg_generator.ingest.loader import Document

logger = logging.getLogger(__name__)


def normalize_vietnamese_unicode(text: str) -> str:
    """Return NFC text while removing duplicate combining marks in a glyph."""
    characters: list[str] = []
    combining_marks: set[str] = set()
    for character in unicodedata.normalize("NFD", text):
        if unicodedata.combining(character) == 0:
            combining_marks.clear()
        elif character in combining_marks:
            continue
        else:
            combining_marks.add(character)
        characters.append(character)
    return unicodedata.normalize("NFC", "".join(characters))


class TextCleanerBackend(ABC):
    """Abstract backend for language-specific text cleaning."""

    @abstractmethod
    def clean(self, text: str) -> str:
        ...


class EnglishCleaner(TextCleanerBackend):
    """English-specific text normalization."""

    def clean(self, text: str) -> str:
        text = text.strip()
        # Normalize line endings while preserving paragraph and line structure.
        # Quality filtering uses line boundaries to identify copied boilerplate.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\t\f\v ]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalize quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        # Normalize dashes
        text = text.replace("\u2013", "-").replace("\u2014", "--")
        # Normalize ellipsis
        text = text.replace("\u2026", "...")
        # Strip non-printable characters (keep newlines)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text


class VietnameseCleaner(TextCleanerBackend):
    """Vietnamese normalization that preserves original readable surface text."""

    def clean(self, text: str) -> str:
        return normalize_vietnamese_unicode(EnglishCleaner().clean(text))


class TextCleaner:
    """Cleans and normalizes raw text documents."""

    def __init__(self, language: Language = Language.ENGLISH) -> None:
        self.backend = self._get_backend(language)

    @staticmethod
    def _get_backend(language: Language) -> TextCleanerBackend:
        if language == Language.VIETNAMESE:
            return VietnameseCleaner()
        return EnglishCleaner()

    def clean(self, document: Document) -> Document:
        """Clean a single document in place."""
        document.content = self.backend.clean(document.content)
        return document

    def clean_batch(self, documents: list[Document]) -> list[Document]:
        """Clean a batch of documents."""
        cleaned = []
        for doc in documents:
            if doc.content.strip():
                cleaned.append(self.clean(doc))
        logger.info(f"Cleaned {len(cleaned)} documents (filtered {len(documents) - len(cleaned)} empty)")
        return cleaned
