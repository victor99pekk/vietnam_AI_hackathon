"""Near-duplicate detection using MinHash, SimHash, and n-gram approaches."""

import hashlib
import logging
from typing import Callable

from datasketch import MinHash, MinHashLSH
from kg_generator.ingest.loader import Document

logger = logging.getLogger(__name__)

# Number of hash functions for MinHash (higher = more accurate, slower)
MINHASH_PERMUTATIONS = 128
LSH_THRESHOLD_DEFAULT = 0.85


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
            return self._minhash_dedup(documents)
        elif self.method == "simhash":
            return self._simhash_dedup(documents)
        elif self.method == "ngram":
            return self._ngram_dedup(documents)
        else:
            logger.warning(f"Unknown dedup method '{self.method}', falling back to exact match")
            return self._exact_dedup(documents)

    def _minhash_dedup(self, documents: list[Document]) -> list[Document]:
        """MinHash + LSH for scalable near-duplicate detection."""
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        minhashes: list[tuple[MinHash, Document]] = []
        for i, doc in enumerate(documents):
            m = self._make_minhash(doc.content)
            lsh.insert(f"doc_{i}", m)
            minhashes.append((m, doc))

        # Group duplicates
        clusters: dict[int, list[int]] = {}
        for i, (m, _) in enumerate(minhashes):
            results = lsh.query(m)
            # Find the canonical cluster for this set of results
            rep = min(int(r.split("_")[1]) for r in results)
            if rep not in clusters:
                clusters[rep] = []
            clusters[rep].append(i)

        duplicate_indices: set[int] = set()
        for cluster_indices in clusters.values():
            if len(cluster_indices) > 1:
                # Keep the first, mark rest as duplicates
                for idx in cluster_indices[1:]:
                    duplicate_indices.add(idx)

        kept = [doc for i, doc in enumerate(documents) if i not in duplicate_indices]
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

    def _make_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for token in self._tokenize(text):
            m.update(token.encode("utf-8"))
        return m

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
