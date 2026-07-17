"""Heuristic-based quality filtering of documents."""

import logging
import re

from kg_generator.ingest.loader import Document

logger = logging.getLogger(__name__)

# Reasonable thresholds for quality filtering
MIN_CHAR_LENGTH = 50
MIN_WORD_COUNT = 10
MAX_SYMBOL_RATIO = 0.4
MAX_REPETITION_RATIO = 0.3


class QualityFilter:
    """Filters out low-quality documents using heuristic rules."""

    def __init__(
        self,
        min_chars: int = MIN_CHAR_LENGTH,
        min_words: int = MIN_WORD_COUNT,
        max_symbol_ratio: float = MAX_SYMBOL_RATIO,
        max_rep_ratio: float = MAX_REPETITION_RATIO,
    ) -> None:
        self.min_chars = min_chars
        self.min_words = min_words
        self.max_symbol_ratio = max_symbol_ratio
        self.max_rep_ratio = max_rep_ratio

    def filter(self, documents: list[Document]) -> list[Document]:
        """Filter documents, keeping only those that pass all quality checks."""
        kept: list[Document] = []
        for doc in documents:
            if self._is_quality(doc.content):
                kept.append(doc)
        removed = len(documents) - len(kept)
        if removed:
            logger.info(f"Quality filter: removed {removed} low-quality documents")
        return kept

    def _is_quality(self, text: str) -> bool:
        """Check if text passes all quality heuristics."""
        if len(text) < self.min_chars:
            return False

        words = text.split()
        if len(words) < self.min_words:
            return False

        # Ratio of non-alphanumeric characters
        symbol_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if symbol_count / max(len(text), 1) > self.max_symbol_ratio:
            return False

        # Repetition check — lines that repeat too many times
        lines = text.splitlines()
        if len(lines) > 1:
            line_counts: dict[str, int] = {}
            for line in lines:
                stripped = line.strip()
                if len(stripped) > 5:
                    line_counts[stripped] = line_counts.get(stripped, 0) + 1
            if line_counts:
                max_rep = max(line_counts.values())
                if max_rep / len(lines) > self.max_rep_ratio:
                    return False

        # Gibberish heuristic: too many short "words" of random chars
        alpha_tokens = [w for w in words if w.isalpha()]
        if alpha_tokens:
            short_ratio = sum(1 for w in alpha_tokens if len(w) <= 2) / len(alpha_tokens)
            if short_ratio > 0.6 and len(alpha_tokens) > 20:
                return False

        return True

    def score(self, text: str) -> float:
        """Return a quality score between 0 and 1 (higher = better quality)."""
        if not text:
            return 0.0

        scores: list[float] = []

        # Length score
        scores.append(min(len(text) / 200, 1.0))

        # Word count score
        word_count = len(text.split())
        scores.append(min(word_count / 30, 1.0))

        # Symbol ratio penalty
        symbol_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        scores.append(max(0, 1 - symbol_ratio / self.max_symbol_ratio))

        return sum(scores) / len(scores)
