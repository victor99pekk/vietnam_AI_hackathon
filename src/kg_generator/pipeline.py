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
from kg_generator.extract.entities import Entity, EntityExtractor, EnglishExtractor, VietnameseExtractor
from kg_generator.extract.relations import RelationExtractor
from kg_generator.graph.builder import GraphBuilder
from kg_generator.graph.enrich import GraphEnricher
from kg_generator.ingest.chunker import TextChunker
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
        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

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

    @staticmethod
    def _snip_description(text: str, entity_name: str, context_chars: int = 120) -> str:
        """Extract a brief snippet from text around an entity name for auto-description."""
        idx = text.lower().find(entity_name.lower())
        if idx == -1:
            return ""
        start = max(0, idx - context_chars // 2)
        end = min(len(text), idx + len(entity_name) + context_chars // 2)
        snippet = text[start:end].strip()
        return snippet

    def execute(self) -> None:
        """Run all stages in sequence."""
        logger.info("Starting KG generation pipeline...")

        # ── Stage 1: Ingest & Clean ──
        logger.info("[1/5] Ingesting & cleaning data...")
        documents = self.loader.load(self.config.input_paths)
        documents = self.cleaner.clean_batch(documents)
        logger.info(f"  Loaded & cleaned {len(documents)} documents")

        # ── Chunk ──
        if self.config.chunk_size > 0:
            documents = self.chunker.chunk(documents)
            logger.info(f"  Chunked into {len(documents)} pieces")

        # ── Dedup ──
        logger.info("[2/5] Deduplication & quality filtering...")
        # Quality filter first (lightweight), then dedup (heavier)
        documents = self.quality_filter.filter(documents)
        documents = self.deduplicator.deduplicate(documents)
        logger.info(f"  {len(documents)} documents after dedup")

        # ── Stage 2: Extract Entities & Relations ──
        logger.info("[3/5] Extracting entities & relations...")
        all_triples: list[tuple[str, str, str, str]] = []
        all_entities: list[dict[str, Any]] = []

        # Create Chunk entities for all documents upfront (needed for :NEXT edges
        # that reference future chunks)
        chunk_ids: list[str] = []
        for doc in documents:
            chunk_id = doc.doc_id or doc.source
            chunk_ids.append(chunk_id)
            doc_entity = Entity(
                name=chunk_id,
                label="Chunk",
                mentions=[chunk_id],
                source=doc.source,
                description=doc.content[:200],
                attributes={
                    "text": doc.content,
                    "tokenCount": doc.metadata.get("token_count", len(doc.content.split())),
                    "index": doc.metadata.get("chunk_index", 0),
                    "embedding": None,
                    "source": doc.source,
                    "char_length": len(doc.content),
                    "parent_source": doc.metadata.get("parent_source", doc.source),
                },
            )
            d = doc_entity.to_dict()
            # Inject chunk-specific properties that aren't in the generic entity dict
            d["text"] = doc.content
            d["tokenCount"] = doc.metadata.get("token_count", len(doc.content.split()))
            d["index"] = doc.metadata.get("chunk_index", 0)
            all_entities.append(d)

        # :NEXT edges between consecutive chunks of the same document
        for i in range(len(chunk_ids) - 1):
            # Only link chunks from the same parent document
            if chunk_ids[i].rsplit(":", 1)[0] == chunk_ids[i + 1].rsplit(":", 1)[0]:
                all_triples.append((chunk_ids[i], "NEXT", chunk_ids[i + 1], ""))

        # Create Document nodes (one per source file) and :PART_OF edges
        seen_docs: set[str] = set()
        for doc in documents:
            parent = doc.metadata.get("parent_source", doc.source)
            parent_name = Path(parent).name if parent else "unknown"
            if parent_name not in seen_docs:
                seen_docs.add(parent_name)
                doc_node = Entity(
                    name=parent_name,
                    label="Document",
                    mentions=[parent_name],
                    source=parent,
                    description=f"Source document: {parent}",
                    attributes={
                        "source": parent,
                        "chunk_count": sum(
                            1 for d in documents
                            if d.metadata.get("parent_source", d.source) == parent
                        ),
                    },
                )
                d = doc_node.to_dict()
                d["chunk_count"] = sum(
                    1 for doc in documents
                    if doc.metadata.get("parent_source", doc.source) == parent
                )
                all_entities.append(d)

        for doc in documents:
            parent = doc.metadata.get("parent_source", doc.source)
            parent_name = Path(parent).name if parent else "unknown"
            chunk_id = doc.doc_id or doc.source
            all_triples.append((chunk_id, "PART_OF", parent_name, ""))

        # Now extract entities and :MENTIONS edges from each chunk
        for doc in documents:
            entities = self.entity_extractor.extract(doc.content)
            triples = self.relation_extractor.extract(doc.content, entities)

            chunk_id = doc.doc_id or doc.source
            for e in entities:
                # Enrich entity with GraphRAG provenance and description
                e.source = chunk_id
                if not e.description:
                    # Auto-generate a brief description from surrounding text
                    e.description = self._snip_description(doc.content, e.name)
                triples.append((chunk_id, "MENTIONS", e.name, doc.content[:500]))

            all_entities.extend(e.to_dict() for e in entities)
            all_triples.extend(triples)

        logger.info(
            f"  Extracted {len(all_entities)} entities "
            f"(incl. {len(documents)} chunks), {len(all_triples)} triples"
        )

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
