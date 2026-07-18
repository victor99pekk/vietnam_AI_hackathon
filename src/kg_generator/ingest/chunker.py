"""Document chunking — splits long documents into smaller overlapping chunks for fine-grained Graph RAG."""

import logging
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from kg_generator.config import Language
from kg_generator.ingest.loader import Document

logger = logging.getLogger(__name__)

# Sensible defaults (150–300 words ≈ 500–1000 chars for English)
DEFAULT_CHUNK_SIZE = 500   # characters
DEFAULT_CHUNK_OVERLAP = 100  # characters


def count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting (no NLP deps needed)."""
    return len(text.split())


def _sentence_units(text: str, language: Language) -> list[str]:
    """Split readable text into sentences without exposing tokenizer underscores."""
    if language == Language.VIETNAMESE:
        try:
            from underthesea import sent_tokenize

            units = [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
            if units:
                return units
        except ImportError:
            # GraphGen Vietnamese runs intentionally do not require Underthesea.
            pass
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?…])\s+|[\r\n]+", text)
        if part.strip()
    ]


class TextChunker:
    """Splits documents into smaller chunks with configurable size and overlap."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separator: str = "\n\n",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks, preserving metadata."""
        chunks: list[Document] = []
        for doc in documents:
            chunks.extend(self._chunk_one(doc))
        logger.info(
            f"Chunked {len(documents)} documents into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def chunk_count(self, text: str) -> int:
        """Return how many chunks a given text would produce (without actually chunking)."""
        if len(text) <= self.chunk_size:
            return 1
        sections = text.split(self.separator)
        count = 0
        current = ""
        for section in sections:
            if len(current) + len(section) <= self.chunk_size:
                current = (current + self.separator + section).strip(self.separator)
            else:
                if current.strip():
                    count += 1
                current = section
                while len(current) > self.chunk_size:
                    count += 1
                    current = current[self.chunk_size - self.chunk_overlap:]
        if current.strip():
            count += 1
        return max(count, 1)

    def _chunk_one(self, doc: Document) -> list[Document]:
        """Split a single document into chunks."""
        text = doc.content

        # If document is shorter than chunk_size, keep as-is
        if len(text) <= self.chunk_size:
            doc.metadata["chunk_index"] = 0
            doc.metadata["token_count"] = count_tokens(text)
            return [doc]

        chunks: list[Document] = []

        sections = text.split(self.separator)
        current = ""
        for section in sections:
            if len(current) + len(section) <= self.chunk_size:
                current = (current + self.separator + section).strip(self.separator)
            else:
                if current.strip():
                    chunks.append(self._make_chunk(doc, current.strip(), len(chunks)))
                if self.chunk_overlap > 0 and current:
                    overlap_text = current[-self.chunk_overlap:].lstrip()
                    current = overlap_text + self.separator + section
                else:
                    current = section
                while len(current) > self.chunk_size:
                    chunks.append(self._make_chunk(doc, current[:self.chunk_size], len(chunks)))
                    current = current[self.chunk_size - self.chunk_overlap:]

        if current.strip():
            chunks.append(self._make_chunk(doc, current.strip(), len(chunks)))

        return chunks

    def _make_chunk(self, doc: Document, text: str, idx: int) -> Document:
        """Create a Document for a GraphRAG-ready Chunk node.

        Node properties: id, text, tokenCount, index, embedding (placeholder).
        """
        from pathlib import Path
        parent_name = Path(doc.source).stem if doc.source else "doc"
        chunk_id = f"{parent_name}:chunk{idx}"

        return Document(
            content=text,
            source=doc.source,
            doc_id=chunk_id,
            metadata={
                **doc.metadata,
                "chunk_index": idx,
                "token_count": count_tokens(text),
                "char_length": len(text),
                "parent_source": doc.source,
                "parent_doc_id": doc.doc_id,
            },
        )


class SentenceChunker(TextChunker):
    """Pack complete sentences into token-budgeted, language-aware chunks."""

    def __init__(
        self,
        target_tokens: int = 450,
        overlap_tokens: int = 60,
        language: Language = Language.ENGLISH,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be greater than zero")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be between zero and target_tokens")
        super().__init__(chunk_size=target_tokens, chunk_overlap=overlap_tokens)
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.language = language

    def chunk_count(self, text: str) -> int:
        probe = Document(content=text)
        return len(self._chunk_one(probe))

    def _chunk_one(self, doc: Document) -> list[Document]:
        units = self._expanded_units(doc.content)
        if not units:
            return []
        packed = self._pack_units(units)
        if len(packed) == 1:
            doc.metadata["chunk_index"] = 0
            doc.metadata["token_count"] = count_tokens(doc.content)
            return [doc]
        return [self._make_chunk(doc, text, index) for index, text in enumerate(packed)]

    def _expanded_units(self, text: str) -> list[str]:
        expanded: list[str] = []
        for sentence in _sentence_units(text, self.language):
            words = sentence.split()
            if len(words) <= self.target_tokens:
                expanded.append(sentence)
                continue
            # A single over-budget sentence is split on word boundaries.
            start = 0
            while start < len(words):
                expanded.append(" ".join(words[start:start + self.target_tokens]))
                start += self.target_tokens
        return expanded

    def _pack_units(self, units: Sequence[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for unit in units:
            unit_tokens = count_tokens(unit)
            if current and current_tokens + unit_tokens > self.target_tokens:
                chunks.append(" ".join(current))
                current = self._overlap_units(current)
                current_tokens = sum(count_tokens(item) for item in current)
                if current and current_tokens + unit_tokens > self.target_tokens:
                    current = []
                    current_tokens = 0
            current.append(unit)
            current_tokens += unit_tokens
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _overlap_units(self, units: Sequence[str]) -> list[str]:
        if not self.overlap_tokens:
            return []
        selected: list[str] = []
        tokens = 0
        for unit in reversed(units):
            unit_tokens = count_tokens(unit)
            if tokens + unit_tokens > self.overlap_tokens:
                break
            selected.append(unit)
            tokens += unit_tokens
        return list(reversed(selected))


class SemanticChunker(SentenceChunker):
    """Create sentence-aligned chunks at multilingual semantic topic shifts."""

    def __init__(
        self,
        target_tokens: int = 450,
        overlap_tokens: int = 60,
        language: Language = Language.ENGLISH,
        similarity_threshold: float = 0.55,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        encoder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between zero and one")
        super().__init__(target_tokens, overlap_tokens, language)
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.encoder = encoder
        self._embedder: Any | None = None

    def _chunk_one(self, doc: Document) -> list[Document]:
        units = self._expanded_units(doc.content)
        if not units:
            return []
        if len(units) == 1:
            doc.metadata["chunk_index"] = 0
            doc.metadata["token_count"] = count_tokens(doc.content)
            return [doc]
        embeddings = self._encode(units)
        packed = self._pack_semantic_units(units, embeddings)
        if len(packed) == 1:
            doc.metadata["chunk_index"] = 0
            doc.metadata["token_count"] = count_tokens(doc.content)
            return [doc]
        return [self._make_chunk(doc, text, index) for index, text in enumerate(packed)]

    def _encode(self, texts: list[str]) -> Sequence[Sequence[float]]:
        if self.encoder is not None:
            return self.encoder(texts)
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Semantic chunking requires sentence-transformers. "
                    "Install it with: uv sync --extra embeddings"
                ) from error
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _pack_semantic_units(
        self,
        units: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> list[str]:
        if len(embeddings) != len(units):
            raise ValueError("Semantic encoder returned the wrong number of embeddings")
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        minimum_topic_tokens = max(1, self.target_tokens // 3)

        for index, unit in enumerate(units):
            unit_tokens = count_tokens(unit)
            topic_shift = (
                index > 0
                and current_tokens >= minimum_topic_tokens
                and self._cosine(embeddings[index - 1], embeddings[index])
                < self.similarity_threshold
            )
            over_budget = bool(current) and current_tokens + unit_tokens > self.target_tokens
            if current and (topic_shift or over_budget):
                chunks.append(" ".join(current))
                current = self._overlap_units(current)
                current_tokens = sum(count_tokens(item) for item in current)
                if current and current_tokens + unit_tokens > self.target_tokens:
                    current = []
                    current_tokens = 0
            current.append(unit)
            current_tokens += unit_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
        dot = sum(float(left) * float(right) for left, right in zip(first, second))
        first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
        second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
        return dot / max(first_norm * second_norm, 1e-12)
