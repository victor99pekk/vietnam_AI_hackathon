"""Heuristic-based quality filtering of documents."""

import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from kg_generator.ingest.loader import Document

logger = logging.getLogger(__name__)

# Reasonable thresholds for quality filtering
MIN_CHAR_LENGTH = 50
MIN_WORD_COUNT = 10
MAX_SYMBOL_RATIO = 0.4
MAX_REPETITION_RATIO = 0.3


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds shared by curation and KG input filtering."""

    min_chars: int = MIN_CHAR_LENGTH
    min_words: int = MIN_WORD_COUNT
    max_symbol_ratio: float = MAX_SYMBOL_RATIO
    max_repeated_line_ratio: float = MAX_REPETITION_RATIO
    max_short_token_ratio: float = 0.6


@dataclass(frozen=True)
class QualityProfile:
    """Explainable document-quality result for audits and filtering."""

    score: float
    char_count: int
    word_count: int
    symbol_ratio: float
    repeated_line_ratio: float
    short_token_ratio: float | None
    rejection_reasons: tuple[str, ...]
    review_flags: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        """True when a document is kept automatically."""
        return not self.rejection_reasons

    @property
    def requires_review(self) -> bool:
        """True when a kept document has suspicious signals for human review."""
        return bool(self.review_flags)

    @property
    def reasons(self) -> tuple[str, ...]:
        """Backward-compatible combined view of hard rejections and review flags."""
        return self.rejection_reasons + self.review_flags

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["rejection_reasons"] = list(self.rejection_reasons)
        result["review_flags"] = list(self.review_flags)
        result["reasons"] = list(self.reasons)
        result["accepted"] = self.accepted
        result["requires_review"] = self.requires_review
        return result


class QualityProfiler:
    """Language-agnostic, explainable quality profiling."""

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def profile(
        self,
        text: str,
        language: str = "en",
        tokens: Sequence[str] | None = None,
    ) -> QualityProfile:
        """Profile text, optionally using language-aware word tokens.

        Curation supplies Vietnamese word segmentation here. Existing KG
        callers keep the dependency-free whitespace-token fallback.
        """
        rejection_reasons: list[str] = []
        review_flags: list[str] = []
        char_count = len(text)
        words = list(tokens) if tokens is not None else re.findall(r"\S+", text, flags=re.UNICODE)
        word_count = len(words)
        symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
        symbol_ratio = symbols / max(char_count, 1)
        meaningful_lines = [line.strip() for line in text.splitlines() if line.strip()]
        repeated_line_ratio = (
            max(meaningful_lines.count(line) for line in set(meaningful_lines)) / len(meaningful_lines)
            if len(meaningful_lines) > 1 else 0.0
        )
        alphabetic_tokens = [token for token in words if token.isalpha()]
        short_token_ratio = (
            sum(1 for token in alphabetic_tokens if len(token) <= 2) / len(alphabetic_tokens)
            if alphabetic_tokens else 0.0
        )
        if not text.strip():
            rejection_reasons.append("empty_content")
        if "\ufffd" in text:
            rejection_reasons.append("invalid_unicode_replacement")
        if char_count < self.thresholds.min_chars:
            rejection_reasons.append("too_short_characters")
        if word_count < self.thresholds.min_words:
            rejection_reasons.append("too_short_words")
        if symbol_ratio > self.thresholds.max_symbol_ratio:
            review_flags.append("excessive_symbols")
        if repeated_line_ratio > self.thresholds.max_repeated_line_ratio and len(meaningful_lines) > 1:
            review_flags.append("repeated_lines")
        if re.search(r"(.)\1{9,}", text, flags=re.DOTALL):
            review_flags.append("repeated_characters")
        # Vietnamese uses short syllables as normal words (e.g. "và", "của", "là"),
        # so this English-oriented heuristic must not be used as a quality signal for vi.
        if language != "vi" and len(alphabetic_tokens) > 20 and short_token_ratio > self.thresholds.max_short_token_ratio:
            review_flags.append("short_token_gibberish")
        score = sum((
            min(char_count / 200, 1.0),
            min(word_count / 30, 1.0),
            max(0.0, 1 - symbol_ratio / max(self.thresholds.max_symbol_ratio, 0.001)),
            max(0.0, 1 - repeated_line_ratio),
        )) / 4
        return QualityProfile(
            score,
            char_count,
            word_count,
            symbol_ratio,
            repeated_line_ratio,
            short_token_ratio if language != "vi" else None,
            tuple(rejection_reasons),
            tuple(review_flags),
        )


class QualityFilter:
    """Filters out low-quality documents using heuristic rules."""

    def __init__(
        self,
        min_chars: int = MIN_CHAR_LENGTH,
        min_words: int = MIN_WORD_COUNT,
        max_symbol_ratio: float = MAX_SYMBOL_RATIO,
        max_rep_ratio: float = MAX_REPETITION_RATIO,
        language: str = "en",
    ) -> None:
        self.min_chars = min_chars
        self.min_words = min_words
        self.max_symbol_ratio = max_symbol_ratio
        self.max_rep_ratio = max_rep_ratio
        self.language = language
        self.profiler = QualityProfiler(QualityThresholds(
            min_chars=min_chars,
            min_words=min_words,
            max_symbol_ratio=max_symbol_ratio,
            max_repeated_line_ratio=max_rep_ratio,
        ))

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
        return self.profiler.profile(text, language=self.language).accepted

    def score(self, text: str) -> float:
        """Return a quality score between 0 and 1 (higher = better quality)."""
        return self.profiler.profile(text, language=self.language).score
