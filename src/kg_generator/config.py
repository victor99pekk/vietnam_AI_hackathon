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
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Dedup
    dedup_threshold: float = 0.85
    dedup_method: str = "minhash"  # minhash, simhash, ngram

    # Extract
    llm_model: str = "deepseek-chat"
    use_llm: bool = False
    spacy_model: str = "en_core_web_sm"

    # Resolve (string matching is safer than embedding-based for short/ID-like names)
    resolve_threshold: float = 0.85
    resolve_method: str = "string"  # string | embedding

    # Neo4j — configured via environment variables only (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""

    # Export
    export_formats: list[str] = field(default_factory=lambda: ["json", "graphml"])

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        pipeline = data.get("pipeline", {})
        return cls(
            language=Language(pipeline.get("language", "en")),
            graph_backend=GraphBackend(pipeline.get("graph_backend", "networkx")),
            input_paths=[Path(p) for p in pipeline.get("input_paths", [])],
            file_formats=pipeline.get("file_formats", ["txt", "json", "csv"]),
            dedup_threshold=pipeline.get("dedup_threshold", 0.85),
            dedup_method=pipeline.get("dedup_method", "minhash"),
            llm_model=pipeline.get("llm_model", "gpt-4o-mini"),
            use_llm=pipeline.get("use_llm", False),
            spacy_model=pipeline.get("spacy_model", "en_core_web_sm"),
            resolve_threshold=pipeline.get("resolve_threshold", 0.85),
            resolve_method=pipeline.get("resolve_method", "string"),
            export_formats=pipeline.get("export_formats", ["json", "graphml"]),
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

    return config
