"""Near-duplicate detection using MinHash, SimHash, and n-gram approaches."""

import hashlib
import logging
from dataclasses import dataclass
from typing import Sequence

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


class GlobalDeduplicator:
    """Cluster all documents and choose a deterministic canonical member."""

    def __init__(self, threshold: float = LSH_THRESHOLD_DEFAULT, num_perm: int = MINHASH_PERMUTATIONS) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Deduplication threshold must be between 0 and 1.")
        self.threshold = threshold
        self.num_perm = num_perm

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

        exact: dict[str, int] = {}
        signatures: list[object] = []
        for index, record in enumerate(records):
            content = str(record["content"])
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash in exact:
                union(index, exact[content_hash])
            else:
                exact[content_hash] = index
            signatures.append(self._minhash(content))

        if MinHash is not None and MinHashLSH is not None:
            lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            for index, signature in enumerate(signatures):
                lsh.insert(str(index), signature)
            for index, signature in enumerate(signatures):
                for candidate in lsh.query(signature):
                    if int(candidate) < index:
                        union(index, int(candidate))
        else:
            gram_sets = [self._grams(str(record["content"])) for record in records]
            for index, grams in enumerate(gram_sets):
                for candidate_index, candidate_grams in enumerate(gram_sets[:index]):
                    similarity = len(grams & candidate_grams) / max(len(grams | candidate_grams), 1)
                    if similarity >= self.threshold:
                        union(index, candidate_index)

        groups: dict[int, list[int]] = {}
        for index in range(len(records)):
            groups.setdefault(find(index), []).append(index)
        assignments: dict[str, DuplicateAssignment] = {}
        for members in groups.values():
            canonical_index = sorted(members, key=lambda i: (-float(records[i]["quality_score"]), str(records[i]["doc_id"])))[0]
            canonical_id = str(records[canonical_index]["doc_id"])
            cluster_id = f"dup-{min(str(records[i]['doc_id']) for i in members)}" if len(members) > 1 else None
            for index in members:
                record_id = str(records[index]["doc_id"])
                assignments[record_id] = DuplicateAssignment(cluster_id, canonical_id, len(members) > 1 and index != canonical_index)
        return assignments

    def _minhash(self, text: str) -> object:
        if MinHash is None:
            return self._grams(text)
        signature = MinHash(num_perm=self.num_perm)
        for gram in self._grams(text):
            signature.update(gram.encode("utf-8"))
        return signature

    @staticmethod
    def _grams(text: str) -> set[str]:
        return {text[index:index + 3] for index in range(max(len(text) - 2, 1))}


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
