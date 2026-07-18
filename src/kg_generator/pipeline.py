"""Pipeline orchestrator — ties together all stages of KG generation."""

import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from kg_generator.config import PipelineConfig, Language
from kg_generator.dedup.near_dedup import Deduplicator
from kg_generator.dedup.quality import QualityFilter
from kg_generator.evaluate.metrics import QualityEvaluator
from kg_generator.export.exporter import GraphExporter
from kg_generator.extract.entities import Entity, EntityExtractor, EnglishExtractor, VietnameseExtractor
from kg_generator.extract.graphgen import GraphGenExtractor
from kg_generator.extract.relations import RelationExtractor
from kg_generator.graph.builder import GraphBuilder
from kg_generator.graph.enrich import GraphEnricher
from kg_generator.identity import chunk_id as stable_chunk_id
from kg_generator.identity import document_id as stable_document_id
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
        # GraphGen performs joint extraction and must not require an offline
        # Vietnamese NLP installation.
        self.entity_extractor = (
            None if config.use_llm else self._build_entity_extractor()
        )
        self.graphgen_extractor = (
            GraphGenExtractor(
                language=config.language,
                model_name=config.llm_model,
                entity_types=tuple(config.graphgen_entity_types),
                max_gleanings=config.graphgen_max_gleanings,
            )
            if config.use_llm
            else None
        )
        self.relation_extractor = RelationExtractor(
            language=config.language,
            use_llm=False,
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
        all_triples: list[tuple[str, ...]] = []
        structural_entities: list[dict[str, Any]] = []
        extracted_entities: list[dict[str, Any]] = []

        # Create Chunk entities for all documents upfront (needed for :NEXT edges
        # that reference future chunks)
        chunk_contexts: list[tuple[Any, str, str]] = []
        for doc in documents:
            parent_source = doc.metadata.get("parent_source", doc.source)
            source_document_id = doc.metadata.get("parent_doc_id", doc.doc_id)
            parent_id = stable_document_id(parent_source, source_document_id)
            index = doc.metadata.get("chunk_index", 0)
            chunk_id = stable_chunk_id(parent_id, index, doc.content)
            chunk_contexts.append((doc, parent_id, chunk_id))
            parent_name = Path(parent_source).name if parent_source else source_document_id or "unknown"
            doc_entity = Entity(
                name=f"{parent_name} chunk {index}",
                label="Chunk",
                mentions=[],
                source=doc.source,
                node_id=chunk_id,
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
            structural_entities.append(d)

        # :NEXT edges between consecutive chunks of the same document
        for i in range(len(chunk_contexts) - 1):
            _, parent_id, current_chunk_id = chunk_contexts[i]
            _, next_parent_id, next_chunk_id = chunk_contexts[i + 1]
            if parent_id == next_parent_id:
                all_triples.append(
                    (current_chunk_id, "NEXT", next_chunk_id, "", current_chunk_id)
                )

        # Create Document nodes (one per source file) and :PART_OF edges
        seen_docs: set[str] = set()
        for doc, parent_id, _ in chunk_contexts:
            parent = doc.metadata.get("parent_source", doc.source)
            parent_name = Path(parent).name if parent else "unknown"
            if parent_id not in seen_docs:
                seen_docs.add(parent_id)
                doc_node = Entity(
                    name=parent_name,
                    label="Document",
                    mentions=[parent_name],
                    source=parent,
                    node_id=parent_id,
                    description=f"Source document: {parent}",
                    attributes={
                        "source": parent,
                        "chunk_count": sum(
                            1 for d in documents
                            if stable_document_id(
                                d.metadata.get("parent_source", d.source),
                                d.metadata.get("parent_doc_id", d.doc_id),
                            ) == parent_id
                        ),
                    },
                )
                d = doc_node.to_dict()
                d["chunk_count"] = sum(
                    1 for doc in documents
                    if stable_document_id(
                        doc.metadata.get("parent_source", doc.source),
                        doc.metadata.get("parent_doc_id", doc.doc_id),
                    ) == parent_id
                )
                structural_entities.append(d)

        for _, parent_id, chunk_id in chunk_contexts:
            all_triples.append((chunk_id, "PART_OF", parent_id, "", chunk_id))

        # Now extract entities and :MENTIONS edges from each chunk
        for doc, _, chunk_id in chunk_contexts:
            if self.graphgen_extractor is not None:
                entities, triples = self.graphgen_extractor.extract(
                    doc.content, source_chunk_id=chunk_id
                )
            else:
                if self.entity_extractor is None:
                    raise RuntimeError("Baseline entity extractor is not configured")
                entities = self.entity_extractor.extract(doc.content)
                triples = self.relation_extractor.extract(
                    doc.content, entities, source_chunk_id=chunk_id
                )

            for e in entities:
                # Enrich entity with GraphRAG provenance and description
                e.source = chunk_id
                if not e.description:
                    # Auto-generate a brief description from surrounding text
                    e.description = self._snip_description(doc.content, e.name)
                evidence = self.relation_extractor._find_evidence(
                    doc.content, e.name, e.name
                )
                triples.append((chunk_id, "MENTIONS", e.id, evidence, chunk_id))

            extracted_entities.extend(e.to_dict() for e in entities)
            all_triples.extend(triples)

        logger.info(
            f"  Extracted {len(structural_entities) + len(extracted_entities)} entities "
            f"(incl. {len(documents)} chunks), {len(all_triples)} triples"
        )

        # ── Stage 3: Resolve Entities ──
        logger.info("[4/5] Resolving entities...")
        resolved_extracted, entity_id_map = self.resolver.resolve_with_mapping(extracted_entities)
        resolved_entities = structural_entities + resolved_extracted
        all_triples = [
            (
                entity_id_map.get(triple[0], triple[0]),
                triple[1],
                entity_id_map.get(triple[2], triple[2]),
                triple[3] if len(triple) > 3 else "",
                triple[4] if len(triple) > 4 else "",
                *triple[5:],
            )
            for triple in all_triples
        ]
        if self.graphgen_extractor is not None:
            resolved_extracted, all_triples = self.graphgen_extractor.aggregate_descriptions(
                resolved_extracted,
                extracted_entities,
                entity_id_map,
                all_triples,
            )
            resolved_entities = structural_entities + resolved_extracted
        logger.info(f"  {len(resolved_entities)} unique entities after resolution")

        # ── Stage 4: Build Graph ──
        logger.info("[5/5] Building graph...")
        graph = self.graph_builder.build(resolved_entities, all_triples)
        graph = self.enricher.enrich(graph)
        extraction_metadata = self._extraction_metadata()
        graph.graph["language"] = self.config.language.value
        graph.graph["extraction_method"] = extraction_metadata["method"]

        # ── Evaluate ──
        logger.info("Evaluating quality...")
        metrics = self.evaluator.evaluate_graph(graph, resolved_entities, all_triples)
        metrics["extraction"] = {
            **extraction_metadata,
            "max_gleanings": (
                self.config.graphgen_max_gleanings
                if self.graphgen_extractor is not None
                else 0
            ),
        }
        self._log_metrics(metrics)

        # ── Export ──
        logger.info("Exporting...")
        self.exporter.export(
            graph=graph,
            entities=resolved_entities,
            triples=all_triples,
            output_dir=self.output_dir,
            formats=self.config.export_formats,
            metadata={
                "language": self.config.language.value,
                "extraction": extraction_metadata,
            },
        )

        # Save metrics
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info("Pipeline complete!")

    def _extraction_metadata(self) -> dict[str, Any]:
        if self.graphgen_extractor is not None:
            return {
                "language": self.config.language.value,
                "method": "graphgen",
                "backend": "deepseek",
                "model": self.config.llm_model,
                "backend_version": None,
                "prompt_version": self.graphgen_extractor.prompt_version,
            }

        backend = "underthesea" if self.config.language == Language.VIETNAMESE else "spacy"
        try:
            backend_version = version(backend)
        except PackageNotFoundError:
            backend_version = None
        return {
            "language": self.config.language.value,
            "method": "baseline",
            "backend": backend,
            "model": None,
            "backend_version": backend_version,
            "prompt_version": None,
        }
    def execute_neo4j(self, session: Any, *, clear: bool = False) -> dict[str, Any]:
        """Run all stages, writing directly to Neo4j per-chunk.

        Unlike ``execute()``, this method never builds a full in-memory graph.
        Entities are resolved against the existing Neo4j database, and every
        node / edge is written via ``MERGE`` as each chunk is processed.

        Parameters
        ----------
        session:
            An active Neo4j ``Session`` (from ``neo4j.Driver.session()``).
        clear:
            If ``True``, delete all existing nodes and relationships first.

        Returns
        -------
        dict
            Graph statistics queried from Neo4j after completion.
        """
        from kg_generator.graph.neo4j_builder import Neo4jGraphBuilder
        from kg_generator.resolve.neo4j_resolver import Neo4jEntityResolver
        from kg_generator.export.neo4j_upload import replace_documents

        builder = Neo4jGraphBuilder(session, ontology=self.config.ontology)
        resolver = Neo4jEntityResolver(session, threshold=self.config.resolve_threshold)

        if clear:
            builder.clear_database()

        # ── Stage 1: Ingest & Clean ──
        logger.info("[1/4] Ingesting & cleaning data...")
        documents = self.loader.load(self.config.input_paths)
        documents = self.cleaner.clean_batch(documents)
        logger.info(f"  Loaded & cleaned {len(documents)} documents")

        if self.config.chunk_size > 0:
            documents = self.chunker.chunk(documents)
            logger.info(f"  Chunked into {len(documents)} pieces")

        # ── Dedup ──
        logger.info("[2/4] Deduplication & quality filtering...")
        documents = self.quality_filter.filter(documents)
        documents = self.deduplicator.deduplicate(documents)
        logger.info(f"  {len(documents)} documents after dedup")

        # ── Build chunk contexts (same logic as execute()) ──
        chunk_contexts: list[tuple[Any, str, str]] = []
        for doc in documents:
            parent_source = doc.metadata.get("parent_source", doc.source)
            source_document_id = doc.metadata.get("parent_doc_id", doc.doc_id)
            parent_id = stable_document_id(parent_source, source_document_id)
            index = doc.metadata.get("chunk_index", 0)
            chunk_id = stable_chunk_id(parent_id, index, doc.content)
            chunk_contexts.append((doc, parent_id, chunk_id))

        # ── Replace any documents being re-uploaded ──
        all_doc_ids = sorted({pid for _, pid, _ in chunk_contexts})
        if all_doc_ids and not clear:
            replace_documents(session, all_doc_ids)

        # ── Stage 3: Extract & write per-chunk ──
        logger.info("[3/4] Extracting entities & relations → Neo4j...")
        seen_docs: set[str] = set()
        chunk_count_by_doc: dict[str, int] = {}

        # First pass: count chunks per document
        for _, parent_id, _ in chunk_contexts:
            chunk_count_by_doc[parent_id] = chunk_count_by_doc.get(parent_id, 0) + 1

        # Write structural nodes & edges
        for doc, parent_id, chunk_id in chunk_contexts:
            # MERGE Chunk node
            builder.merge_chunk(
                chunk_id,
                source=doc.source,
                text=doc.content,
                token_count=doc.metadata.get("token_count", len(doc.content.split())),
                index=doc.metadata.get("chunk_index", 0),
            )

            # MERGE Document node (once per document)
            if parent_id not in seen_docs:
                seen_docs.add(parent_id)
                parent_source = doc.metadata.get("parent_source", doc.source)
                parent_name = Path(parent_source).name if parent_source else "unknown"
                builder.merge_document(
                    parent_id,
                    name=parent_name,
                    description=f"Source document: {parent_source}",
                    source=parent_source,
                    chunk_count=chunk_count_by_doc.get(parent_id, 0),
                )

            # MERGE PART_OF edge
            builder.merge_structural_edge(chunk_id, parent_id, "PART_OF")

        # Write NEXT edges
        for i in range(len(chunk_contexts) - 1):
            _, p1, c1 = chunk_contexts[i]
            _, p2, c2 = chunk_contexts[i + 1]
            if p1 == p2:
                builder.merge_structural_edge(c1, c2, "NEXT")

        # Second pass: extract entities & relations per chunk
        total_extracted = 0
        total_triples = 0
        for doc, _, chunk_id in chunk_contexts:
            if self.graphgen_extractor is not None:
                entities, triples = self.graphgen_extractor.extract(
                    doc.content, source_chunk_id=chunk_id
                )
            else:
                entities = self.entity_extractor.extract(doc.content)
                triples = self.relation_extractor.extract(
                    doc.content, entities, source_chunk_id=chunk_id
                )

            # Enrich entities
            for e in entities:
                e.source = chunk_id
                if not e.description:
                    e.description = self._snip_description(doc.content, e.name)

            # Resolve entities against Neo4j
            entity_dicts = [e.to_dict() for e in entities]
            resolved_dicts, id_map = resolver.resolve_with_mapping(entity_dicts)

            # Write MENTIONS edges (chunk → resolved entity)
            for entity_dict in resolved_dicts:
                eid = entity_dict.get("id", "")
                evidence = self.relation_extractor._find_evidence(
                    doc.content, entity_dict.get("name", ""), entity_dict.get("name", "")
                )
                builder.merge_structural_edge(chunk_id, eid, "MENTIONS")

            # Write extracted relation edges (entity → entity)
            for triple in triples:
                subj = id_map.get(triple[0], triple[0])
                pred = triple[1]
                obj = id_map.get(triple[2], triple[2])
                evidence = triple[3] if len(triple) > 3 else ""
                desc = triple[5] if len(triple) > 5 else ""
                builder.merge_edge(
                    subj, obj, pred,
                    evidence_sentence=evidence,
                    source_chunk_id=chunk_id,
                    description=desc,
                )

            total_extracted += len(resolved_dicts)
            total_triples += len(triples)

        logger.info(
            f"  Processed {len(chunk_contexts)} chunks: "
            f"{total_extracted} entities, {total_triples} triples"
        )

        # ── Stage 4: Post-processing ──
        logger.info("[4/4] Post-processing...")
        builder.compute_pagerank()
        stats = builder.compute_stats()

        # Save metrics
        metrics: dict[str, Any] = {
            **stats,
            "extraction": {
                "method": "graphgen" if self.graphgen_extractor is not None else "baseline",
                "model": self.config.llm_model if self.graphgen_extractor is not None else None,
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Neo4j pipeline complete!  Stats: %s", stats)
        return metrics

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        logger.info("  Quality Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"    {key}: {value:.4f}")
            else:
                logger.info(f"    {key}: {value}")
