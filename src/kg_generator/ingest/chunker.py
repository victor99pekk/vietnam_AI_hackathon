"""Document chunking — splits long documents into smaller overlapping chunks for fine-grained Graph RAG."""

import logging

from kg_generator.ingest.loader import Document

logger = logging.getLogger(__name__)

# Sensible defaults (150–300 words ≈ 500–1000 chars for English)
DEFAULT_CHUNK_SIZE = 500   # characters
DEFAULT_CHUNK_OVERLAP = 100  # characters


def count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting (no NLP deps needed)."""
    return len(text.split())


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
