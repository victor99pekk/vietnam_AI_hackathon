"""Near-duplicate detection using MinHash, SimHash, and n-gram approaches."""

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Callable, Sequence

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:  # pragma: no cover - used only in minimal environments
    MinHash = None  # type: ignore[assignment,misc]
    MinHashLSH = None  # type: ignore[assignment,misc]
from kg_generator.ingest.loader import Document

logger = logging.getLogger(__name__)

# Number of hash functions for MinHash (higher = more accurate, slower)
MINHASH_PERMUTATIONS = 128
LSH_THRESHOLD_DEFAULT = 0.85


@dataclass(frozen=True)
class DuplicateAssignment:
    """Duplicate decision retained for curation audit records."""

    cluster_id: str | None
    canonical_id: str | None
    is_duplicate: bool
    method: str | None = None
    similarity: float | None = None
    matched_record_id: str | None = None


@dataclass(frozen=True)
class DuplicateMatch:
    """Direct evidence that connected two records in a duplicate cluster."""

    record_id: str
    matched_record_id: str
    method: str
    similarity: float


class GlobalDeduplicator:
    """Cluster exact and surface-level near duplicates with MinHash candidates.

    ``shingle_fn`` keeps the historic character-trigram behaviour by default,
    while allowing curation to supply language-aware word shingles without
    changing the KG pipeline.
    """

    def __init__(
        self,
        threshold: float = LSH_THRESHOLD_DEFAULT,
        num_perm: int = MINHASH_PERMUTATIONS,
        shingle_fn: Callable[[str], set[str]] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Deduplication threshold must be between 0 and 1.")
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_fn = shingle_fn or self._grams
        self.last_matches: list[DuplicateMatch] = []

    def cluster(self, records: Sequence[dict[str, object]]) -> dict[str, DuplicateAssignment]:
        parent = list(range(len(records)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(first_root, second_root)

        matches: dict[tuple[int, int], DuplicateMatch] = {}
        exact: dict[str, int] = {}
        signatures: list[object] = []
        gram_sets: list[set[str]] = []
        for index, record in enumerate(records):
            content = str(record["content"])
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            grams = self.shingle_fn(content)
            gram_sets.append(grams)
            if content_hash in exact:
                existing = exact[content_hash]
                union(index, existing)
                self._record_match(matches, records, index, existing, "exact_hash", 1.0)
            else:
                exact[content_hash] = index
            signatures.append(self._minhash(grams))

        if MinHash is not None and MinHashLSH is not None:
            lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            for index, signature in enumerate(signatures):
                lsh.insert(str(index), signature)
            for index, signature in enumerate(signatures):
                for candidate in lsh.query(signature):
                    candidate_index = int(candidate)
                    if candidate_index < index:
                        similarity = self._jaccard(gram_sets[index], gram_sets[candidate_index])
                        if similarity >= self.threshold:
                            union(index, candidate_index)
                            self._record_match(matches, records, index, candidate_index, "minhash_jaccard", similarity)
        else:
            for index, grams in enumerate(gram_sets):
                for candidate_index, candidate_grams in enumerate(gram_sets[:index]):
                    similarity = len(grams & candidate_grams) / max(len(grams | candidate_grams), 1)
                    if similarity >= self.threshold:
                        union(index, candidate_index)
                        self._record_match(matches, records, index, candidate_index, "ngram_jaccard_fallback", similarity)

        groups: dict[int, list[int]] = {}
        for index in range(len(records)):
            groups.setdefault(find(index), []).append(index)
        self.last_matches = sorted(matches.values(), key=lambda match: (match.record_id, match.matched_record_id, match.method))
        return self._assignments(records, groups, self.last_matches)

    def _minhash(self, shingles: set[str]) -> object:
        if MinHash is None:
            return shingles
        signature = MinHash(num_perm=self.num_perm)
        for gram in shingles:
            signature.update(gram.encode("utf-8"))
        return signature

    @staticmethod
    def _grams(text: str) -> set[str]:
        return {text[index:index + 3] for index in range(max(len(text) - 2, 1))}

    @staticmethod
    def _jaccard(first: set[str], second: set[str]) -> float:
        return len(first & second) / max(len(first | second), 1)

    @staticmethod
    def _record_match(
        matches: dict[tuple[int, int], DuplicateMatch],
        records: Sequence[dict[str, object]],
        first: int,
        second: int,
        method: str,
        similarity: float,
    ) -> None:
        key = tuple(sorted((first, second)))
        current = matches.get(key)
        if current and current.similarity >= similarity:
            return
        matches[key] = DuplicateMatch(
            record_id=str(records[first]["doc_id"]),
            matched_record_id=str(records[second]["doc_id"]),
            method=method,
            similarity=similarity,
        )

    @staticmethod
    def _assignments(
        records: Sequence[dict[str, object]],
        groups: dict[int, list[int]],
        matches: Sequence[DuplicateMatch],
    ) -> dict[str, DuplicateAssignment]:
        assignments: dict[str, DuplicateAssignment] = {}
        matches_by_record: dict[str, list[DuplicateMatch]] = {}
        for match in matches:
            matches_by_record.setdefault(match.record_id, []).append(match)
            matches_by_record.setdefault(match.matched_record_id, []).append(match)
        for members in groups.values():
            canonical_index = sorted(members, key=lambda i: (-float(records[i]["quality_score"]), str(records[i]["doc_id"])))[0]
            canonical_id = str(records[canonical_index]["doc_id"])
            cluster_id = f"dup-{min(str(records[i]['doc_id']) for i in members)}" if len(members) > 1 else None
            for index in members:
                record_id = str(records[index]["doc_id"])
                evidence = sorted(matches_by_record.get(record_id, []), key=lambda match: (-match.similarity, match.method, match.matched_record_id))[0:1]
                assignments[record_id] = DuplicateAssignment(
                    cluster_id=cluster_id,
                    canonical_id=canonical_id,
                    is_duplicate=len(members) > 1 and index != canonical_index,
                    method=evidence[0].method if evidence else None,
                    similarity=evidence[0].similarity if evidence else None,
                    matched_record_id=(
                        evidence[0].matched_record_id if evidence and evidence[0].record_id == record_id
                        else evidence[0].record_id if evidence else None
                    ),
                )
        return assignments


class SemanticDeduplicator:
    """Embedding-based semantic duplicate clustering for small and medium datasets.

    This is an opt-in semantic baseline inspired by SemDeDup. It evaluates all
    document pairs, so it is intentionally capped and is not a web-scale ANN
    implementation.
    """

    def __init__(
        self,
        threshold: float = 0.92,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        max_records: int = 5_000,
        encoder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Semantic deduplication threshold must be between 0 and 1.")
        self.threshold = threshold
        self.model_name = model_name
        self.max_records = max_records
        self.encoder = encoder
        self.last_matches: list[DuplicateMatch] = []

    def cluster(self, records: Sequence[dict[str, object]]) -> dict[str, DuplicateAssignment]:
        if len(records) > self.max_records:
            raise ValueError(
                f"Semantic deduplication supports at most {self.max_records} records per run; "
                "use MinHash or add an ANN index for larger datasets."
            )
        parent = list(range(len(records)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(first_root, second_root)

        embeddings = self._encode([str(record["content"]) for record in records])
        matches: list[DuplicateMatch] = []
        for index, embedding in enumerate(embeddings):
            for candidate_index, candidate_embedding in enumerate(embeddings[:index]):
                similarity = self._cosine_similarity(embedding, candidate_embedding)
                if similarity >= self.threshold:
                    union(index, candidate_index)
                    matches.append(DuplicateMatch(
                        record_id=str(records[index]["doc_id"]),
                        matched_record_id=str(records[candidate_index]["doc_id"]),
                        method="semantic_cosine",
                        similarity=similarity,
                    ))
        groups: dict[int, list[int]] = {}
        for index in range(len(records)):
            groups.setdefault(find(index), []).append(index)
        self.last_matches = sorted(matches, key=lambda match: (match.record_id, match.matched_record_id))
        return GlobalDeduplicator._assignments(records, groups, self.last_matches)

    def _encode(self, texts: list[str]) -> Sequence[Sequence[float]]:
        if self.encoder:
            return self.encoder(texts)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Semantic deduplication requires sentence-transformers. Install it with: "
                "uv pip install -e '.[embeddings]'"
            ) from error
        model = SentenceTransformer(self.model_name)
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def _cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
        dot_product = sum(left * right for left, right in zip(first, second))
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        return dot_product / max(first_norm * second_norm, 1e-12)


class Deduplicator:
    """Detects and removes near-duplicate documents."""

    def __init__(
        self,
        threshold: float = LSH_THRESHOLD_DEFAULT,
        method: str = "minhash",
        num_perm: int = MINHASH_PERMUTATIONS,
    ) -> None:
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.method = method
        self.num_perm = num_perm

    def deduplicate(self, documents: list[Document]) -> list[Document]:
        """Return deduplicated list of documents."""
        if not documents:
            return []

        if self.method == "minhash":
            return self._global_minhash_dedup(documents)
        elif self.method == "simhash":
            return self._simhash_dedup(documents)
        elif self.method == "ngram":
            return self._ngram_dedup(documents)
        else:
            logger.warning(f"Unknown dedup method '{self.method}', falling back to exact match")
            return self._exact_dedup(documents)

    def _global_minhash_dedup(self, documents: list[Document]) -> list[Document]:
        """Use the audit-capable global MinHash engine used by curation."""
        records = [
            {"doc_id": str(index), "content": document.content, "quality_score": 0.0}
            for index, document in enumerate(documents)
        ]
        assignments = GlobalDeduplicator(self.threshold, self.num_perm).cluster(records)
        kept = [
            document for index, document in enumerate(documents)
            if not assignments[str(index)].is_duplicate
        ]
        removed = len(documents) - len(kept)
        if removed:
            logger.info(f"MinHash dedup: removed {removed} near-duplicate documents")
        return kept

    def _simhash_dedup(self, documents: list[Document]) -> list[Document]:
        """SimHash-based deduplication (simpler, less sensitive to small changes)."""
        seen: list[int] = []
        kept: list[Document] = []

        for doc in documents:
            sig = self._simhash(doc.content)
            is_dup = False
            for existing in seen:
                if self._hamming_distance(sig, existing) <= self._simhash_distance_threshold():
                    is_dup = True
                    break
            if not is_dup:
                seen.append(sig)
                kept.append(doc)

        removed = len(documents) - len(kept)
        if removed:
            logger.info(f"SimHash dedup: removed {removed} near-duplicate documents")
        return kept

    def _ngram_dedup(self, documents: list[Document]) -> list[Document]:
        """Jaccard similarity over character n-grams (no external deps)."""
        kept: list[Document] = []
        seen_sets: list[set[str]] = []

        for doc in documents:
            ngrams = self._char_ngrams(doc.content, n=5)
            is_dup = False
            for existing in seen_sets:
                jaccard = len(ngrams & existing) / max(len(ngrams | existing), 1)
                if jaccard >= self.threshold:
                    is_dup = True
                    break
            if not is_dup:
                seen_sets.append(ngrams)
                kept.append(doc)

        removed = len(documents) - len(kept)
        if removed:
            logger.info(f"N-gram dedup: removed {removed} near-duplicate documents")
        return kept

    def _exact_dedup(self, documents: list[Document]) -> list[Document]:
        """Fallback: exact hash-based duplicate removal."""
        seen: set[str] = set()
        kept: list[Document] = []
        for doc in documents:
            h = hashlib.sha256(doc.content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                kept.append(doc)
        removed = len(documents) - len(kept)
        if removed:
            logger.info(f"Exact dedup: removed {removed} exact duplicate documents")
        return kept

    # --- helpers ---

    def _tokenize(self, text: str, n: int = 3) -> list[str]:
        """Character n-gram tokenization (language-agnostic)."""
        return [text[i : i + n] for i in range(max(len(text) - n + 1, 1))]

    def _char_ngrams(self, text: str, n: int = 5) -> set[str]:
        return set(text[i : i + n] for i in range(max(len(text) - n + 1, 1)))

    def _simhash(self, text: str) -> int:
        """Compute a 64-bit SimHash fingerprint."""
        tokens = self._tokenize(text, n=4)
        v = [0] * 64
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for j in range(64):
                if h & (1 << j):
                    v[j] += 1
                else:
                    v[j] -= 1
        return sum((1 << i) for i in range(64) if v[i] > 0)

    @staticmethod
    def _hamming_distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def _simhash_distance_threshold(self) -> int:
        """Convert Jaccard threshold to approximate Hamming distance."""
        # Rough mapping: jaccard 0.85 -> ~3 bits
        if self.threshold >= 0.95:
            return 2
        elif self.threshold >= 0.85:
            return 3
        elif self.threshold >= 0.75:
            return 4
        return 6
