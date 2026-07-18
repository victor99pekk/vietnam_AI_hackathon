"""MongoDB document archive — permanent store for all document versions.

Every document version is archived here *before* it touches the knowledge graph.
This makes MongoDB the ground truth: if the KG gets corrupted, it can be rebuilt
from the archive.  The ``mongo_uri`` config flag controls whether this layer is
active — when empty, the pipeline runs without MongoDB.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_generator.ingest.loader import Document

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MongoDocumentStore:
    """Archives every document version and tracks the KG lifecycle.

    Collections
    -----------
    ``documents``
        One record per unique (canonical_id × content_hash) combination.
        Version numbers are auto-incremented per ``canonical_id``.

    ``archived_chunks``
        Chunk nodes that were removed from the KG when a newer document
        version replaced them.  Preserved for auditing and rollback.

    ``ingestion_runs``
        High-level pipeline run records for provenance tracking.
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "kg_documents",
    ) -> None:
        try:
            from pymongo import ASCENDING, DESCENDING, IndexModel, MongoClient
            from pymongo.collection import Collection
        except ImportError as exc:
            raise RuntimeError(
                "pymongo is required for MongoDB integration. "
                "Install with: uv sync --extra mongo"
            ) from exc

        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.documents: Collection = self.db.documents
        self.archived_chunks: Collection = self.db.archived_chunks
        self.ingestion_runs: Collection = self.db.ingestion_runs
        self._ensure_indexes()
        logger.info("Connected to MongoDB: %s / %s", uri, database)

    # ── Indexes ─────────────────────────────────────────────────

    def _ensure_indexes(self) -> None:
        from pymongo import ASCENDING, DESCENDING, IndexModel

        self.documents.create_indexes([
            IndexModel(
                [("canonical_id", ASCENDING), ("version", DESCENDING)],
                name="idx_canonical_version",
            ),
            IndexModel(
                [("content_hash", ASCENDING)],
                unique=True,
                name="idx_content_hash",
            ),
            IndexModel(
                [("upload_date", DESCENDING)],
                name="idx_upload_date",
            ),
        ])

        self.archived_chunks.create_indexes([
            IndexModel(
                [("canonical_id", ASCENDING)],
                name="idx_archived_canonical",
            ),
            IndexModel(
                [("chunk_id", ASCENDING)],
                name="idx_archived_chunk",
            ),
            IndexModel(
                [("archived_at", DESCENDING)],
                name="idx_archived_at",
            ),
        ])

        self.ingestion_runs.create_indexes([
            IndexModel(
                [("run_id", ASCENDING)],
                unique=True,
                name="idx_run_id",
            ),
            IndexModel(
                [("started_at", DESCENDING)],
                name="idx_started_at",
            ),
        ])

    # ── Document lifecycle ──────────────────────────────────────

    def document_exists(self, canonical_id: str) -> bool:
        """Return ``True`` if a document with this canonical id was ever ingested."""
        return self.documents.count_documents(
            {"canonical_id": canonical_id}, limit=1
        ) > 0

    def get_latest_version(self, canonical_id: str) -> dict[str, Any] | None:
        """Return the most recent version of a document, or ``None``."""
        return self.documents.find_one(
            {"canonical_id": canonical_id},
            sort=[("version", DESCENDING)],
        )

    def store_document(
        self,
        doc: Document,
        canonical_id: str,
        *,
        upload_date: str = "",
    ) -> str:
        """Archive a document version.  Returns the content hash.

        If the exact same content was already stored (same ``content_hash``)
        the upload timestamp is updated but no new version is created.
        """
        content_hash = _safe_hash(doc.content)

        latest = self.get_latest_version(canonical_id)
        version = (latest["version"] + 1) if latest else 1
        parent_hash = latest["content_hash"] if latest else None

        # Same content → update metadata only, no new version
        if latest and latest["content_hash"] == content_hash:
            self.documents.update_one(
                {"content_hash": content_hash},
                {"$set": {"last_upload_date": upload_date or _now()}},
            )
            logger.debug(
                "Document %s unchanged (hash=%s…) — metadata updated",
                canonical_id, content_hash[:12],
            )
            return content_hash

        record = {
            "canonical_id": canonical_id,
            "content_hash": content_hash,
            "content_preview": doc.content[:200],
            "content_full": doc.content,
            "source": doc.source,
            "upload_date": upload_date or _now(),
            "version": version,
            "parent_hash": parent_hash,
            "title": doc.metadata.get("title", ""),
            "url": doc.metadata.get("url", ""),
            "token_count": len(doc.content.split()),
            "char_length": len(doc.content),
            "language": doc.metadata.get("language", "unknown"),
            "kg_chunk_ids": [],
            "kg_triple_count": 0,
            "metadata": doc.metadata,
        }

        try:
            self.documents.insert_one(record)
            logger.info(
                "Stored %s v%d (hash=%s…, %d tokens)",
                canonical_id, version, content_hash[:12], record["token_count"],
            )
        except Exception:
            # DuplicateKeyError on content_hash — race condition
            logger.debug("Content hash %s… already present", content_hash[:12])

        return content_hash

    # ── Chunk archival (when a document is replaced in the KG) ──

    def archive_chunks(
        self,
        canonical_id: str,
        chunks: list[dict[str, Any]],
        replaced_version: int,
    ) -> int:
        """Archive old chunks before their document is replaced in the KG.

        Returns the number of chunks archived.
        """
        if not chunks:
            return 0

        now = _now()
        docs = [
            {
                "canonical_id": canonical_id,
                "chunk_id": c.get("id", ""),
                "chunk_text": c.get("text", c.get("description", ""))[:500],
                "chunk_index": c.get("index", 0),
                "kg_node": c,
                "replaced_by_version": replaced_version + 1,
                "archived_at": now,
            }
            for c in chunks
        ]
        self.archived_chunks.insert_many(docs)
        logger.info(
            "Archived %d chunks for %s (replaced v%d)",
            len(chunks), canonical_id, replaced_version,
        )
        return len(chunks)

    # ── KG linkage ──────────────────────────────────────────────

    def link_kg_chunks(
        self,
        canonical_id: str,
        chunk_ids: list[str],
        triple_count: int,
    ) -> None:
        """Record which KG chunks were produced from this document version."""
        self.documents.update_one(
            {"canonical_id": canonical_id},
            {"$set": {
                "kg_chunk_ids": chunk_ids,
                "kg_triple_count": triple_count,
            }},
            sort=[("version", DESCENDING)],
        )

    # ── Ingestion run tracking ──────────────────────────────────

    def start_run(self, run_id: str, *, config: dict[str, Any] | None = None) -> None:
        self.ingestion_runs.insert_one({
            "run_id": run_id,
            "started_at": _now(),
            "config": config or {},
            "documents_processed": 0,
            "documents_replaced": 0,
            "documents_new": 0,
            "documents_unchanged": 0,
            "status": "running",
        })

    def complete_run(self, run_id: str, stats: dict[str, int]) -> None:
        self.ingestion_runs.update_one(
            {"run_id": run_id},
            {"$set": {
                "status": "completed",
                "completed_at": _now(),
                **stats,
            }},
        )

    def close(self) -> None:
        self.client.close()
        logger.debug("MongoDB connection closed")
