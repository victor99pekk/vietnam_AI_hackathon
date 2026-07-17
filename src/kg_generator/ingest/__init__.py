"""Stage 1: Data ingestion — loading and cleaning raw text."""

from kg_generator.ingest.cleaner import TextCleaner
from kg_generator.ingest.loader import DataLoader, Document

__all__ = ["DataLoader", "Document", "TextCleaner"]
