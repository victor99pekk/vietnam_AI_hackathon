"""Language-aware normalization, record splitting, and semantic review helpers."""

from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kg_generator.dedup.near_dedup import DuplicateMatch


DEFAULT_BGE_MODEL = "BAAI/bge-m3"


@dataclass(frozen=True)
class TextSpan:
    """A half-open character span in normalized source text."""

    start: int
    end: int


class CurationTextProcessor:
    """Normalize and segment one declared corpus language at a time."""

    def __init__(self, language: str) -> None:
        if language not in {"en", "vi"}:
            raise ValueError("Curation language must be 'en' or 'vi'.")
        self.language = language
        self._english_nlp: Any | None = None

    def normalize(self, text: str) -> str:
        """Repair safe Unicode issues without changing linguistic content."""
        text = unicodedata.normalize("NFC", text)
        try:
            from ftfy import fix_text

            text = fix_text(text)
        except ImportError:
            # The base package remains useful for audits. The curation extra
            # installs ftfy for production runs.
            pass
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(
            character
            for character in text
            if not (
                unicodedata.category(character) == "Cc"
                and character not in {"\n", "\t"}
            )
        )
        text = re.sub(r"[\t\f\v ]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def word_tokens(self, text: str) -> list[str]:
        """Return words for quality checks and MinHash shingles."""
        if self.language == "vi":
            try:
                from underthesea import word_tokenize
            except ImportError as error:
                raise RuntimeError(
                    "Vietnamese curation requires underthesea. Install: uv pip install -e '.[curation]'"
                ) from error
            segmented = word_tokenize(text, format="text")
            return [token for token in segmented.split() if token]
        return [
            token.text
            for token in self._english_doc(text)
            if not token.is_space and not token.is_punct
        ]

    def word_shingles(self, text: str, size: int = 5) -> set[str]:
        """Case-insensitive word shingles for language-aware MinHash."""
        tokens = [token.casefold() for token in self.word_tokens(text)]
        if not tokens:
            return {""}
        if len(tokens) < size:
            return {"\x1f".join(tokens)}
        return {"\x1f".join(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}

    def sentence_spans(self, text: str) -> list[TextSpan]:
        """Return sentence spans, with a safe character-preserving fallback."""
        if not text:
            return []
        if self.language == "en":
            spans = [TextSpan(sentence.start_char, sentence.end_char) for sentence in self._english_doc(text).sents]
            return spans or [TextSpan(0, len(text))]

        try:
            from underthesea import sent_tokenize
        except ImportError as error:
            raise RuntimeError(
                "Vietnamese curation requires underthesea. Install: uv pip install -e '.[curation]'"
            ) from error
        cursor = 0
        spans: list[TextSpan] = []
        for sentence in sent_tokenize(text):
            candidate = str(sentence).strip()
            if not candidate:
                continue
            start = text.find(candidate, cursor)
            if start < 0:
                return self._fallback_sentence_spans(text)
            end = start + len(candidate)
            spans.append(TextSpan(start, end))
            cursor = end
        return spans or self._fallback_sentence_spans(text)

    def _english_doc(self, text: str) -> Any:
        if self._english_nlp is None:
            try:
                import spacy
            except ImportError as error:  # pragma: no cover - base dependency
                raise RuntimeError("English curation requires spaCy.") from error
            self._english_nlp = spacy.blank("en")
            self._english_nlp.add_pipe("sentencizer")
        return self._english_nlp(text)

    @staticmethod
    def _fallback_sentence_spans(text: str) -> list[TextSpan]:
        spans: list[TextSpan] = []
        start = 0
        for match in re.finditer(r"(?<=[.!?…])\s+", text):
            end = match.start()
            if end > start:
                spans.append(TextSpan(start, end))
            start = match.end()
        if start < len(text):
            spans.append(TextSpan(start, len(text)))
        return spans or [TextSpan(0, len(text))]


class BgeTokenCounter:
    """Count model tokens using the same tokenizer as semantic review."""

    def __init__(self, model_name: str = DEFAULT_BGE_MODEL, revision: str | None = None) -> None:
        self.model_name = model_name
        self.revision = revision
        self._tokenizer: Any | None = None

    def count(self, text: str) -> int:
        tokenizer = self._load()
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    @property
    def resolved_revision(self) -> str | None:
        tokenizer = self._load()
        return getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or self.revision

    def _load(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "BGE tokenization requires transformers. Install: uv pip install -e '.[curation]'"
                ) from error
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.revision,
                use_fast=True,
            )
        return self._tokenizer


def split_text_to_token_limit(
    text: str,
    processor: CurationTextProcessor,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[tuple[TextSpan, int]]:
    """Pack complete sentences, splitting only an oversized sentence safely."""
    if max_tokens < 3:
        raise ValueError("max_record_tokens must leave room for tokenizer special tokens.")
    if not text:
        return []
    starts = sorted({span.start for span in processor.sentence_spans(text) if span.start > 0})
    boundaries = [0, *starts, len(text)]
    units: list[TextSpan] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if start < end:
            units.extend(_split_oversized_span(text, TextSpan(start, end), count_tokens, max_tokens))

    results: list[tuple[TextSpan, int]] = []
    current_start: int | None = None
    current_end = 0
    for unit in units:
        if current_start is None:
            current_start, current_end = unit.start, unit.end
            continue
        candidate = text[current_start:unit.end]
        if count_tokens(candidate) <= max_tokens:
            current_end = unit.end
            continue
        current_text = text[current_start:current_end]
        results.append((TextSpan(current_start, current_end), count_tokens(current_text)))
        current_start, current_end = unit.start, unit.end
    if current_start is not None:
        current_text = text[current_start:current_end]
        results.append((TextSpan(current_start, current_end), count_tokens(current_text)))
    return results


def _split_oversized_span(
    text: str,
    span: TextSpan,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[TextSpan]:
    if count_tokens(text[span.start:span.end]) <= max_tokens:
        return [span]
    pieces: list[TextSpan] = []
    start = span.start
    while start < span.end:
        low, high = start + 1, span.end
        best = start
        while low <= high:
            midpoint = (low + high) // 2
            if count_tokens(text[start:midpoint]) <= max_tokens:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best == start:
            raise ValueError("A single tokenizer unit exceeds max_record_tokens.")
        word_break = text.rfind(" ", start + 1, best)
        if word_break > start:
            best = word_break + 1
        pieces.append(TextSpan(start, best))
        start = best
    return pieces


class SemanticReviewer:
    """Batched BGE-M3 embedding and non-destructive semantic duplicate review."""

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str | None,
        device: str,
        threshold: float,
        top_k: int,
        batch_token_budget: int,
        encoder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.threshold = threshold
        self.top_k = top_k
        self.batch_token_budget = batch_token_budget
        self.encoder = encoder
        self._model: Any | None = None
        self.effective_batch_sizes: list[int] = []

    def review(self, records: Sequence[dict[str, Any]], cache_path: Path | None = None) -> list[DuplicateMatch]:
        if len(records) < 2:
            return []
        embeddings = self._embeddings(records, cache_path)
        scores, indices = self._nearest_neighbors(embeddings)
        matches: list[DuplicateMatch] = []
        seen: set[tuple[int, int]] = set()
        for index, row in enumerate(indices):
            for rank, candidate in enumerate(row):
                candidate_index = int(candidate)
                if candidate_index < 0 or candidate_index == index:
                    continue
                similarity = float(scores[index][rank])
                if similarity < self.threshold:
                    continue
                key = tuple(sorted((index, candidate_index)))
                if key in seen:
                    continue
                seen.add(key)
                matches.append(DuplicateMatch(
                    record_id=str(records[index]["record_id"]),
                    matched_record_id=str(records[candidate_index]["record_id"]),
                    method="semantic_cosine",
                    similarity=similarity,
                ))
        return sorted(matches, key=lambda match: (match.record_id, match.matched_record_id))

    def _embeddings(self, records: Sequence[dict[str, Any]], cache_path: Path | None) -> Any:
        import numpy as np

        cached = self._load_cache(cache_path)
        hashes_by_id = {
            str(record["record_id"]): str(record["content_hash"])
            for record in records
        }
        embeddings_by_id: dict[str, Any] = {
            record_id: vector
            for record_id, (content_hash, vector) in cached.items()
            if hashes_by_id.get(record_id) == content_hash
        }
        pending = [record for record in records if str(record["record_id"]) not in embeddings_by_id]
        pending.sort(key=lambda record: (-int(record["token_count"]), str(record["record_id"])))
        position = 0
        while position < len(pending):
            batch: list[dict[str, Any]] = []
            batch_tokens = 0
            while position < len(pending) and len(batch) < 32:
                record = pending[position]
                token_count = int(record["token_count"])
                if batch and batch_tokens + token_count > self.batch_token_budget:
                    break
                batch.append(record)
                batch_tokens += token_count
                position += 1
            if not batch:
                batch.append(pending[position])
                position += 1
            for completed, vectors in self._encode_with_retry(batch):
                for record, vector in zip(completed, vectors):
                    embeddings_by_id[str(record["record_id"])] = vector
                self._save_cache(cache_path, records, embeddings_by_id)
        result = np.asarray([embeddings_by_id[str(record["record_id"])] for record in records], dtype="float32")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-12)

    def _encode_with_retry(self, batch: list[dict[str, Any]]) -> Iterable[tuple[list[dict[str, Any]], Any]]:
        try:
            vectors = self._encode([str(record["text"]) for record in batch])
            self.effective_batch_sizes.append(len(batch))
            yield batch, vectors
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or len(batch) == 1:
                raise
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - sentence-transformers installs torch
                pass
            midpoint = len(batch) // 2
            yield from self._encode_with_retry(batch[:midpoint])
            yield from self._encode_with_retry(batch[midpoint:])

    def _encode(self, texts: list[str]) -> Sequence[Sequence[float]]:
        if self.encoder is not None:
            return self.encoder(texts)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Semantic review requires sentence-transformers. Install: uv pip install -e '.[curation]'"
            ) from error
        if self.device.startswith("cuda"):
            try:
                import torch
            except ImportError as error:  # pragma: no cover - sentence-transformers installs torch
                raise RuntimeError("CUDA semantic review requires PyTorch.") from error
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but no CUDA device is available. Use --device cpu explicitly.")
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                revision=self.model_revision,
                device=self.device,
            )
        return self._model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _nearest_neighbors(self, embeddings: Any) -> tuple[Any, Any]:
        import numpy as np

        limit = min(self.top_k + 1, len(embeddings))
        try:
            import faiss
        except ImportError as error:
            if self.encoder is None:
                raise RuntimeError(
                    "Semantic review requires faiss. Install: uv pip install -e '.[curation]'"
                ) from error
            scores = embeddings @ embeddings.T
            indices = np.argsort(-scores, axis=1)[:, :limit]
            return np.take_along_axis(scores, indices, axis=1), indices
        index: Any = faiss.IndexFlatIP(embeddings.shape[1])
        if self.device.startswith("cuda") and hasattr(faiss, "StandardGpuResources"):
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        index.add(embeddings)
        return index.search(embeddings, limit)

    def _load_cache(self, cache_path: Path | None) -> dict[str, tuple[str, Any]]:
        if cache_path is None or not cache_path.exists():
            return {}
        import numpy as np

        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["model"][0]) != self._cache_model_key():
                    return {}
                return {
                    str(record_id): (str(content_hash), vector)
                    for record_id, content_hash, vector in zip(
                        cached["record_ids"], cached["content_hashes"], cached["embeddings"]
                    )
                }
        except (OSError, KeyError, ValueError):
            return {}

    def _save_cache(
        self,
        cache_path: Path | None,
        records: Sequence[dict[str, Any]],
        embeddings_by_id: dict[str, Any],
    ) -> None:
        if cache_path is None:
            return
        import numpy as np

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stored = [record for record in records if str(record["record_id"]) in embeddings_by_id]
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            np.savez(
                handle,
                model=np.asarray([self._cache_model_key()]),
                record_ids=np.asarray([str(record["record_id"]) for record in stored]),
                content_hashes=np.asarray([str(record["content_hash"]) for record in stored]),
                embeddings=np.asarray([embeddings_by_id[str(record["record_id"])] for record in stored], dtype="float32"),
            )
        temporary_path.replace(cache_path)

    def _cache_model_key(self) -> str:
        return json.dumps({"model": self.model_name, "revision": self.model_revision}, sort_keys=True)
