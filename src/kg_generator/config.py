"""Knowledge Graph Generator — core configuration and ontology definitions."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Language(str, Enum):
    ENGLISH = "en"
    VIETNAMESE = "vi"


class GraphBackend(str, Enum):
    NETWORKX = "networkx"
    NEO4J = "neo4j"


DEFAULT_GRAPHGEN_ENTITY_TYPES = [
    "concept",
    "date",
    "location",
    "keyword",
    "organization",
    "person",
    "event",
    "work",
    "nature",
    "artificial",
    "science",
    "technology",
    "mission",
    "gene",
]


@dataclass
class Ontology:
    """Defines the schema for a knowledge graph: entity types, relations, attributes."""

    entity_types: dict[str, dict[str, str]] = field(default_factory=dict)
    relationship_types: dict[str, dict[str, str]] = field(default_factory=dict)
    attributes: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "Ontology":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            entity_types=data.get("entity_types", {}),
            relationship_types=data.get("relationship_types", {}),
            attributes=data.get("attributes", {}),
        )


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    language: Language = Language.ENGLISH
    graph_backend: GraphBackend = GraphBackend.NETWORKX
    ontology: Ontology | None = None

    # Ingest
    input_paths: list[Path] = field(default_factory=list)
    file_formats: list[str] = field(default_factory=lambda: ["txt", "json", "csv"])

    # Chunking (0 = disabled, keep documents as-is)
    chunk_method: str = "fixed"  # none | fixed | sentence | semantic
    chunk_size: int = 500
    chunk_overlap: int = 100
    chunk_target_tokens: int = 450
    chunk_overlap_tokens: int = 60
    semantic_chunk_threshold: float = 0.55
    semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Quality and deduplication. ``dedup_method`` remains the backward-
    # compatible chunk-level selector used by existing YAML files.
    quality_method: str = "heuristic"  # none | heuristic
    document_dedup_method: str = "none"
    dedup_threshold: float = 0.85
    dedup_method: str = "minhash"  # none | exact | minhash | simhash | ngram | semantic | layered
    document_dedup_threshold: float = 0.85
    semantic_dedup_threshold: float = 0.92
    semantic_dedup_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    semantic_dedup_max_records: int = 5_000

    # Extract
    llm_model: str = "deepseek-v4-flash"
    use_llm: bool = False
    spacy_model: str = "en_core_web_sm"
    graphgen_entity_types: list[str] = field(
        default_factory=lambda: list(DEFAULT_GRAPHGEN_ENTITY_TYPES)
    )
    graphgen_max_gleanings: int = 3

    # Resolve (string matching is safer than embedding-based for short/ID-like names)
    resolve_threshold: float = 0.85
    resolve_method: str = "string"  # string | embedding
    resolve_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Neo4j — configured via environment variables only (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""

    # MongoDB — document archive (empty = disabled; set to enable versioning)
    mongo_uri: str = ""
    mongo_database: str = "kg_documents"

    # Export
    export_formats: list[str] = field(default_factory=lambda: ["json", "graphml"])

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        pipeline = data.get("pipeline", {})
        chunking = pipeline.get("chunking", {}) or {}
        quality = pipeline.get("quality", {}) or {}
        deduplication = pipeline.get("deduplication", {}) or {}
        extraction = pipeline.get("extraction", {}) or {}
        resolution = pipeline.get("resolution", {}) or {}

        extraction_method = extraction.get("method")
        if extraction_method not in {None, "offline", "graphgen"}:
            raise ValueError("extraction.method must be one of: offline, graphgen")
        use_llm = (
            extraction_method == "graphgen"
            if extraction_method is not None
            else pipeline.get("use_llm", False)
        )
        resolve_method = resolution.get(
            "method", pipeline.get("resolve_method", "string")
        )
        if resolve_method == "string_similarity":
            resolve_method = "string"
        return cls(
            language=Language(pipeline.get("language", "en")),
            graph_backend=GraphBackend(pipeline.get("graph_backend", "networkx")),
            input_paths=[Path(p) for p in pipeline.get("input_paths", [])],
            file_formats=pipeline.get("file_formats", ["txt", "json", "csv"]),
            chunk_method=chunking.get("method", pipeline.get("chunk_method", "fixed")),
            chunk_size=chunking.get("size_chars", pipeline.get("chunk_size", 500)),
            chunk_overlap=chunking.get("overlap_chars", pipeline.get("chunk_overlap", 100)),
            chunk_target_tokens=chunking.get(
                "target_tokens", pipeline.get("chunk_target_tokens", 450)
            ),
            chunk_overlap_tokens=chunking.get(
                "overlap_tokens", pipeline.get("chunk_overlap_tokens", 60)
            ),
            semantic_chunk_threshold=chunking.get(
                "semantic_threshold", pipeline.get("semantic_chunk_threshold", 0.55)
            ),
            semantic_model=chunking.get(
                "embedding_model",
                pipeline.get("semantic_model", "paraphrase-multilingual-MiniLM-L12-v2"),
            ),
            quality_method=quality.get("method", pipeline.get("quality_method", "heuristic")),
            document_dedup_method=deduplication.get(
                "document_method", pipeline.get("document_dedup_method", "none")
            ),
            dedup_method=deduplication.get(
                "chunk_method", pipeline.get("dedup_method", "minhash")
            ),
            dedup_threshold=deduplication.get(
                "chunk_threshold", pipeline.get("dedup_threshold", 0.85)
            ),
            document_dedup_threshold=deduplication.get(
                "document_threshold", pipeline.get("document_dedup_threshold", 0.85)
            ),
            semantic_dedup_threshold=deduplication.get(
                "semantic_threshold", pipeline.get("semantic_dedup_threshold", 0.92)
            ),
            semantic_dedup_model=deduplication.get(
                "embedding_model",
                pipeline.get("semantic_dedup_model", "paraphrase-multilingual-MiniLM-L12-v2"),
            ),
            semantic_dedup_max_records=deduplication.get(
                "max_records", pipeline.get("semantic_dedup_max_records", 5_000)
            ),
            llm_model=extraction.get("model", pipeline.get("llm_model", "deepseek-v4-flash")),
            use_llm=use_llm,
            spacy_model=extraction.get(
                "spacy_model", pipeline.get("spacy_model", "en_core_web_sm")
            ),
            graphgen_entity_types=extraction.get(
                "entity_types", pipeline.get(
                "graphgen_entity_types", list(DEFAULT_GRAPHGEN_ENTITY_TYPES)
                )
            ),
            graphgen_max_gleanings=extraction.get(
                "max_gleanings", pipeline.get("graphgen_max_gleanings", 3)
            ),
            resolve_threshold=resolution.get(
                "threshold", pipeline.get("resolve_threshold", 0.85)
            ),
            resolve_method=resolve_method,
            resolve_model=resolution.get(
                "embedding_model",
                pipeline.get("resolve_model", "paraphrase-multilingual-MiniLM-L12-v2"),
            ),
            export_formats=pipeline.get("export_formats", ["json", "graphml"]),
            mongo_uri=pipeline.get("mongo_uri", ""),
            mongo_database=pipeline.get("mongo_database", "kg_documents"),
        )


def load_config(config_path: Path | None = None) -> PipelineConfig:
    """Load configuration from a YAML file, applying sensible defaults otherwise."""
    if config_path and config_path.exists():
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    # Neo4j connection — only from environment variables
    config.neo4j_uri = os.environ.get("NEO4J_URI", "")
    config.neo4j_user = os.environ.get("NEO4J_USER", "")
    config.neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    # MongoDB — allow env var override (MONGO_URI trumps yaml)
    if os.environ.get("MONGO_URI"):
        config.mongo_uri = os.environ["MONGO_URI"]
    if os.environ.get("MONGO_DATABASE"):
        config.mongo_database = os.environ["MONGO_DATABASE"]

    return config
