"""Pipeline orchestrator — ties together all stages of KG generation."""

import json
import logging
from pathlib import Path
from typing import Any

from kg_generator.config import PipelineConfig, Language
from kg_generator.dedup.near_dedup import Deduplicator
from kg_generator.dedup.quality import QualityFilter
from kg_generator.evaluate.metrics import QualityEvaluator
from kg_generator.export.exporter import GraphExporter
from kg_generator.extract.entities import EntityExtractor, EnglishExtractor, VietnameseExtractor
from kg_generator.extract.relations import RelationExtractor
from kg_generator.graph.builder import GraphBuilder
from kg_generator.graph.enrich import GraphEnricher
from kg_generator.ingest.cleaner import TextCleaner
from kg_generator.ingest.loader import DataLoader
from kg_generator.resolve.resolver import EntityResolver

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full knowledge graph generation pipeline."""

    def __init__(self, config: PipelineConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Stage 1: Ingest ---
        self.loader = DataLoader(config.file_formats)
        self.cleaner = TextCleaner(language=config.language)

        # --- Dedup & Quality ---
        self.deduplicator = Deduplicator(
            threshold=config.dedup_threshold,
            method=config.dedup_method,
        )
        self.quality_filter = QualityFilter(language=config.language.value)

        # --- Stage 2: Extract ---
        self.entity_extractor = self._build_entity_extractor()
        self.relation_extractor = RelationExtractor(
            language=config.language,
            use_llm=config.use_llm,
            model_name=config.llm_model,
        )

        # --- Stage 3: Resolve ---
        self.resolver = EntityResolver(
            threshold=config.resolve_threshold,
            method=config.resolve_method,
        )

        # --- Stage 4: Graph ---
        self.graph_builder = GraphBuilder(
            ontology=config.ontology,
            backend=config.graph_backend,
        )
        self.enricher = GraphEnricher()

        # --- Stage 5: Evaluate & Export ---
        self.evaluator = QualityEvaluator()
        self.exporter = GraphExporter()

    def _build_entity_extractor(self) -> EntityExtractor:
        if self.config.language == Language.VIETNAMESE:
            return VietnameseExtractor()
        return EnglishExtractor(model_name=self.config.spacy_model)

    def execute(self) -> None:
        """Run all stages in sequence."""
        logger.info("Starting KG generation pipeline...")

        # ── Stage 1: Ingest & Clean ──
        logger.info("[1/5] Ingesting & cleaning data...")
        documents = self.loader.load(self.config.input_paths)
        documents = self.cleaner.clean_batch(documents)
        logger.info(f"  Loaded & cleaned {len(documents)} documents")

        # ── Dedup ──
        logger.info("[2/5] Deduplication & quality filtering...")
        # Quality filter first (lightweight), then dedup (heavier)
        documents = self.quality_filter.filter(documents)
        documents = self.deduplicator.deduplicate(documents)
        logger.info(f"  {len(documents)} documents after dedup")

        # ── Stage 2: Extract Entities & Relations ──
        logger.info("[3/5] Extracting entities & relations...")
        all_triples: list[tuple[str, str, str]] = []
        all_entities: list[dict[str, Any]] = []

        for doc in documents:
            entities = self.entity_extractor.extract(doc.content)
            triples = self.relation_extractor.extract(doc.content, entities)
            all_entities.extend(e.to_dict() for e in entities)
            all_triples.extend(triples)

        logger.info(f"  Extracted {len(all_entities)} entities, {len(all_triples)} triples")

        # ── Stage 3: Resolve Entities ──
        logger.info("[4/5] Resolving entities...")
        resolved_entities = self.resolver.resolve(all_entities)
        logger.info(f"  {len(resolved_entities)} unique entities after resolution")

        # ── Stage 4: Build Graph ──
        logger.info("[5/5] Building graph...")
        graph = self.graph_builder.build(resolved_entities, all_triples)
        graph = self.enricher.enrich(graph)

        # ── Evaluate ──
        logger.info("Evaluating quality...")
        metrics = self.evaluator.evaluate_graph(graph, resolved_entities, all_triples)
        self._log_metrics(metrics)

        # ── Export ──
        logger.info("Exporting...")
        self.exporter.export(
            graph=graph,
            entities=resolved_entities,
            triples=all_triples,
            output_dir=self.output_dir,
            formats=self.config.export_formats,
        )

        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Pipeline complete!")

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        logger.info("  Quality Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"    {key}: {value:.4f}")
            else:
                logger.info(f"    {key}: {value}")
