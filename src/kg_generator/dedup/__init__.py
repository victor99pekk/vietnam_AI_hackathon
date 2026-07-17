"""Deduplication and quality filtering stage."""

from kg_generator.dedup.near_dedup import Deduplicator
from kg_generator.dedup.quality import QualityFilter

__all__ = ["Deduplicator", "QualityFilter"]
