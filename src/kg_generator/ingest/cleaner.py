"""Text normalization and cleaning with language-specific backends."""

import re
import logging
from abc import ABC, abstractmethod

from kg_generator.config import Language
from kg_generator.ingest.loader import Document

logger = logging.getLogger(__name__)


class TextCleanerBackend(ABC):
    """Abstract backend for language-specific text cleaning."""

    @abstractmethod
    def clean(self, text: str) -> str:
        ...


class EnglishCleaner(TextCleanerBackend):
    """English-specific text normalization."""

    def clean(self, text: str) -> str:
        text = text.strip()
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
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
    """Vietnamese-specific text normalization. Stub — will use underthesea when vi support is enabled."""

    def clean(self, text: str) -> str:
        # Base cleaning (same as English)
        text = EnglishCleaner().clean(text)
        # TODO: Add underthesea-based tokenization & normalization
        # from underthesea import text_normalize
        # text = text_normalize(text)
        return text


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
