"""Small HTTP API used by the demo frontend.

The API deliberately keeps the browser contract small.  Runs are isolated in a
temporary directory and are exported as JSON by the existing pipeline.
"""

from __future__ import annotations

import re
import tempfile
import os
import hashlib
import threading
import time
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from fastapi.staticfiles import StaticFiles

from kg_generator.config import Language, PipelineConfig
from kg_generator.identity import entity_id

MAX_INPUT_CHARS = 20_000
RUN_LOCK = threading.Lock()
DEEPSEEK_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")

DEMO_TEXT = (
    "Võ Nguyên Giáp sinh năm 1911 tại Quảng Bình. Ông là một nhân vật quan trọng "
    "trong lịch sử Việt Nam và tham gia chỉ huy chiến dịch Điện Biên Phủ năm 1954. "
    "Chiến dịch Điện Biên Phủ góp phần kết thúc Chiến tranh Đông Dương."
)

app = FastAPI(title="AI Việt Knowledge Graph Demo", version="0.1.0")
_SOURCE_DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
DEMO_DIR = Path(os.getenv("KG_DEMO_DIR", "/app/demo"))
if not DEMO_DIR.exists():
    DEMO_DIR = _SOURCE_DEMO_DIR


class PipelineRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    language: Literal["vi", "en"] = "vi"
    extraction: Literal["offline"] = "offline"

    @field_validator("text")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class SourceRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    title: str = "Demo source"
    url: str = ""
    license: str = "Demo"

    @field_validator("text")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source.text must not be blank")
        return value


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    language: Literal["vi", "en"] = "vi"
    extraction: Literal["offline", "graphgen"] = "offline"
    llm_model: Literal["deepseek-v4-flash", "deepseek-v4-pro"] = "deepseek-v4-flash"
    chunk_method: Literal["none", "fixed", "sentence", "semantic"] = "sentence"
    chunk_size: int = Field(500, ge=0, le=10_000)
    chunk_overlap: int = Field(100, ge=0, le=5_000)
    chunk_target_tokens: int = Field(450, ge=1, le=4_000)
    chunk_overlap_tokens: int = Field(60, ge=0, le=3_999)
    semantic_chunk_threshold: float = Field(.55, ge=0, le=1)
    semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    quality_method: Literal["none", "heuristic"] = "heuristic"
    dedup_method: Literal["none", "exact", "minhash", "simhash", "ngram", "semantic", "layered"] = "minhash"
    resolve_method: Literal["string", "embedding"] = "string"
    document_dedup_method: Literal["none", "exact", "minhash", "simhash", "ngram", "semantic", "layered"] = "minhash"
    document_dedup_threshold: float = Field(.85, ge=0, le=1)
    dedup_threshold: float = Field(.85, ge=0, le=1)
    semantic_dedup_threshold: float = Field(.92, ge=0, le=1)
    semantic_dedup_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    semantic_dedup_max_records: int = Field(5_000, ge=1, le=20_000)
    resolve_threshold: float = Field(.85, ge=0, le=1)
    resolve_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    graphgen_max_gleanings: int = Field(3, ge=0, le=5)

    @model_validator(mode="after")
    def validate_relationships(self):
        if self.chunk_method == "fixed" and self.chunk_size > 0 and self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.chunk_method in {"sentence", "semantic"} and self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_target_tokens")
        return self


class RunRequest(BaseModel):
    source: SourceRequest | None = None
    options: RunOptions = Field(default_factory=RunOptions)

    @model_validator(mode="before")
    @classmethod
    def compatibility_shape(cls, values):
        if isinstance(values, dict) and values.get("source") is None and values.get("text"):
            values = dict(values)
            values["source"] = {"text": values.pop("text"), **(values.pop("config", {}) or {})}
        return values

    @field_validator("source")
    @classmethod
    def require_source(cls, value):
        if value is None:
            raise ValueError("source is required")
        return value


def _logical_source(source: SourceRequest) -> dict[str, Any]:
    digest = hashlib.sha256(source.text.encode("utf-8")).hexdigest()[:16]
    return {"id": f"source-{digest}", "title": source.title, "url": source.url, "license": source.license}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/healthz")
def api_healthz() -> dict[str, str]:
    """Public health route; /healthz may be reserved by hosting infrastructure."""
    return {"status": "ok"}


@app.get("/api/demo/sample")
def demo_sample() -> dict[str, str]:
    return {"text": DEMO_TEXT, "language": "vi", "extraction": "offline"}


@app.get("/api/options")
def options() -> dict[str, Any]:
    try:
        import importlib.util
        embeddings_available = importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        embeddings_available = False
    interactive_configured = bool(
        os.getenv("NEO4J_URI")
        and (os.getenv("NEO4J_INTERACTIVE_USER") or os.getenv("NEO4J_USER"))
        and (os.getenv("NEO4J_INTERACTIVE_PASSWORD") or os.getenv("NEO4J_PASSWORD"))
    )
    return {"languages": ["vi", "en"], "extraction": ["offline", "graphgen"],
            "llm_models": list(DEEPSEEK_MODELS), "chunk_methods": ["none", "fixed", "sentence", "semantic"],
            "quality_methods": ["none", "heuristic"], "resolve_methods": ["string", "embedding"],
            "document_dedup_methods": ["none", "exact", "minhash", "simhash", "ngram", "semantic", "layered"],
            "dedup_methods": ["none", "exact", "minhash", "simhash", "ngram", "semantic", "layered"],
            "parameters": {"chunk_target_tokens": {"min": 1, "max": 4000},
                           "chunk_overlap_tokens": {"min": 0, "max": 3999},
                           "threshold": {"min": 0, "max": 1}},
            "availability": {"neo4j": interactive_configured, "interactive_neo4j": interactive_configured,
                             "global_neo4j": bool(os.getenv("NEO4J_URI")),
                             "graphgen": bool(os.getenv("DEEPSEEK_API_KEY")),
                             "embeddings": embeddings_available}}


def _fallback_graph(text: str, language: str) -> dict[str, Any]:
    """Dependency-free extraction fallback for minimal Cloud Run images."""
    # Keep likely proper names and domain terms, while avoiding a blank graph if
    # optional Vietnamese NLP models are unavailable.
    candidates = re.findall(r"[A-ZÀ-ỸĐ][\wÀ-ỹĐđ]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ]*)+", text)
    known = re.findall(r"\b(?:Việt Nam|Điện Biên Phủ|Quảng Bình|Chiến tranh Đông Dương)\b", text)
    names: list[str] = []
    for name in [*candidates, *known]:
        name = name.strip(" ,.;:()")
        if len(name) > 2 and name.casefold() not in {n.casefold() for n in names}:
            names.append(name)
    entities = [
        {
            "id": entity_id("CONCEPT", name),
            "name": name,
            "type": "PERSON" if any(token in name for token in ("Giáp", "Nguyễn")) else "CONCEPT",
            "aliases": [name.casefold()],
            "description": text[:180],
            "confidenceScore": 0.55,
            "importanceScore": 0.0,
            "source": ["demo-input"],
        }
        for name in names
    ]
    triples: list[dict[str, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        present = [entity for entity in entities if entity["name"].casefold() in sentence.casefold()]
        for left, right in zip(present, present[1:]):
            triples.append({
                "subject": left["id"], "predicate": "associated_with", "object": right["id"],
                "evidence_sentence": sentence.strip(), "source_chunk_id": "",
            })
    nodes = [{"id": entity["id"], "name": entity["name"], "type": entity["type"]} for entity in entities]
    return {
        "metadata": {"language": language, "extraction": {"method": "offline-fallback", "backend": "regex"}},
        "graph": {"directed": True, "multigraph": False, "graph": {}, "nodes": nodes,
                  "links": [{"source": t["subject"], "target": t["object"], "predicates": [t["predicate"]]}
                            for t in triples]},
        "entities": entities,
        "triples": triples,
        "stats": {"num_nodes": len(nodes), "num_edges": len(triples), "num_triples": len(triples)},
        "metrics": {"overall_score": 0.0, "extraction": {"method": "offline-fallback"}},
    }


def _run_pipeline(request: PipelineRequest) -> dict[str, Any]:
    # Exercise the repository pipeline by default.  Set this to ``0`` for a
    # minimal image (or while optional numerical dependencies are unavailable).
    if os.getenv("KG_DEMO_USE_FULL_PIPELINE") == "0":
        return _fallback_graph(request.text, request.language)

    # Import lazily: the full evaluation stack can load optional numerical
    # libraries, while health/sample endpoints should remain lightweight.
    from kg_generator.pipeline import Pipeline

    with tempfile.TemporaryDirectory(prefix="kg-demo-") as temp_dir:
        root = Path(temp_dir)
        source = root / "input.txt"
        source.write_text(request.text, encoding="utf-8")
        config = PipelineConfig(
            language=Language(request.language), input_paths=[source], file_formats=["txt"],
            chunk_method="none", chunk_size=0, quality_method="none", dedup_method="none",
            document_dedup_method="none", resolve_method="string", export_formats=["json"],
            use_llm=False,
        )
        Pipeline(config, root / "output").execute()
        output = root / "output" / "knowledge_graph.json"
        payload = json.loads(output.read_text(encoding="utf-8"))
        metrics_path = root / "output" / "metrics.json"
        payload["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        return payload


def _run_nested(request: RunRequest) -> dict[str, Any]:
    source = request.source
    options = request.options
    logical = _logical_source(source)
    if os.getenv("KG_DEMO_USE_FULL_PIPELINE") == "0":
        payload = _fallback_graph(source.text, options.language)
    else:
        from kg_generator.pipeline import Pipeline
        with tempfile.TemporaryDirectory(prefix="kg-run-") as temp_dir:
            root = Path(temp_dir)
            input_path = root / "source.json"
            input_path.write_text(json.dumps({"id": logical["id"], "title": source.title,
                                              "url": source.url, "license": source.license,
                                              "text": source.text}, ensure_ascii=False), encoding="utf-8")
            config = _config_for_run(options, input_path)
            Pipeline(config, root / "output").execute()
            payload = json.loads((root / "output" / "knowledge_graph.json").read_text(encoding="utf-8"))
            metrics_path = root / "output" / "metrics.json"
            payload["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    payload.setdefault("metadata", {})["logical_source"] = logical
    return payload


def _config_for_run(options: RunOptions, input_path: Path) -> PipelineConfig:
    """Translate validated public options to the internal pipeline config."""
    return PipelineConfig(language=Language(options.language), input_paths=[input_path], file_formats=["json"],
        chunk_method=options.chunk_method, chunk_size=options.chunk_size, chunk_overlap=options.chunk_overlap,
        chunk_target_tokens=options.chunk_target_tokens, chunk_overlap_tokens=options.chunk_overlap_tokens,
        semantic_chunk_threshold=options.semantic_chunk_threshold, semantic_model=options.semantic_model,
                quality_method=options.quality_method, dedup_method=options.dedup_method,
                document_dedup_method=options.document_dedup_method, document_dedup_threshold=options.document_dedup_threshold,
                dedup_threshold=options.dedup_threshold, semantic_dedup_threshold=options.semantic_dedup_threshold,
                semantic_dedup_model=options.semantic_dedup_model, semantic_dedup_max_records=options.semantic_dedup_max_records,
                resolve_method=options.resolve_method, resolve_threshold=options.resolve_threshold,
                resolve_model=options.resolve_model, graphgen_max_gleanings=options.graphgen_max_gleanings,
        use_llm=options.extraction == "graphgen", llm_model=options.llm_model, export_formats=["json"])


@app.post("/api/runs")
def create_run(request: RunRequest) -> dict[str, Any]:
    if not RUN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="another pipeline run is in progress")
    started = time.perf_counter()
    try:
        try:
            result = _run_nested(request)
        except Exception as error:
            raise HTTPException(status_code=500, detail={"code": "pipeline_failed", "message": "Pipeline run failed"}) from error
        result["persistence"] = _persist_interactive(result)
        result.setdefault("metadata", {})["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result
    finally:
        RUN_LOCK.release()


def _persist_interactive(payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically replace the interactive graph when Neo4j is configured."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_INTERACTIVE_USER")
    password = os.getenv("NEO4J_INTERACTIVE_PASSWORD") or os.getenv("NEO4J_PASSWORD")
    user = user or os.getenv("NEO4J_USER")
    if not (uri and user and password):
        return {"status": "skipped", "reason": "interactive Neo4j is not configured"}
    try:
        from neo4j import GraphDatabase
        from kg_generator.export.neo4j_upload import replace_documents_atomic
        graph = payload.get("graph", {})
        nodes, edges = graph.get("nodes", []), graph.get("links", graph.get("edges", []))
        document_ids = [n.get("id", "") for n in nodes if n.get("type") == "Document"]
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=os.getenv("NEO4J_INTERACTIVE_DATABASE", "interactive")) as session:
            def writer(tx):
                for node in nodes:
                    label = "Document" if node.get("type") == "Document" else ("Chunk" if node.get("type") == "Chunk" else "Entity")
                    tx.run(f"MERGE (n:{label} {{id:$id}}) SET n.name=$name, n.type=$type, n.description=$description", id=node.get("id", ""), name=node.get("name", ""), type=node.get("type", "Entity"), description=node.get("description", ""))
                for edge in edges:
                    for predicate in edge.get("predicates", [edge.get("label", "related_to")]):
                        tx.run("MATCH (a {id:$source}), (b {id:$target}) MERGE (a)-[r:RELATION {predicate:$predicate}]->(b)", source=edge.get("source"), target=edge.get("target"), predicate=predicate)
            replace_documents_atomic(session, document_ids, writer, clear_all=True)
        driver.close()
        return {"status": "persisted", "database": os.getenv("NEO4J_INTERACTIVE_DATABASE", "interactive")}
    except Exception as error:
        return {"status": "failed", "reason": str(error)}


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRequest) -> dict[str, Any]:
    try:
        return _run_pipeline(request)
    except Exception as error:  # stable public error; details remain server-side
        raise HTTPException(status_code=500, detail="pipeline run failed") from error


def _sample_global_graph() -> dict[str, Any]:
    """Return a bundled, read-only graph when the live database is unavailable."""
    candidates = (
        Path("/app/data/global_sample.json"),
        Path(__file__).resolve().parents[2] / "generated_KGs/output_small/knowledge_graph.json",
    )
    for path in candidates:
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload.setdefault("metadata", {})
                payload["metadata"].update({"source": "sample", "read_only": True, "scope": "global"})
                return payload
        except (OSError, json.JSONDecodeError):
            continue
    return {
        "metadata": {"source": "sample", "read_only": True, "scope": "global"},
        "graph": {"nodes": [{"id": "hcm", "name": "Hồ Chí Minh", "type": "PERSON"},
                              {"id": "vietnam", "name": "Việt Nam", "type": "PLACE"}],
                  "links": [{"source": "hcm", "target": "vietnam", "predicates": ["lãnh đạo"]}]},
        "stats": {"num_nodes": 2, "num_edges": 1, "num_triples": 1},
    }


def _neo4j_graph(*, database: str, user_env: str, password_env: str,
                 node_id: str | None = None, query: str = "", limit: int = 150) -> dict[str, Any] | None:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv(user_env) or os.getenv("NEO4J_USER")
    password = os.getenv(password_env) or os.getenv("NEO4J_PASSWORD")
    if not (uri and user and password):
        return None
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            if node_id:
                record = session.run(
                    "MATCH (n {id:$id}) OPTIONAL MATCH (n)-[r]-(m) "
                    "RETURN collect(DISTINCT n) AS nodes, collect(DISTINCT m) AS neighbors, "
                    "collect(DISTINCT {source:startNode(r).id,target:endNode(r).id,"
                    "predicates:[coalesce(r.predicate,type(r))]}) AS links", id=node_id,
                ).single()
                nodes = (list(record["nodes"] or []) + list(record["neighbors"] or [])) if record else []
                links = list(record["links"] or []) if record else []
            else:
                nodes = [row["node"] for row in session.run(
                    "MATCH (n) WHERE $query='' OR toLower(coalesce(n.name,n.id,'')) CONTAINS toLower($query) "
                    "RETURN n AS node LIMIT $limit", query=query, limit=limit,
                )]
                ids = [str(dict(node).get("id", "")) for node in nodes]
                links = [dict(row) for row in session.run(
                    "MATCH (a)-[r]->(b) WHERE a.id IN $ids AND b.id IN $ids "
                    "RETURN a.id AS source,b.id AS target,[coalesce(r.predicate,type(r))] AS predicates", ids=ids,
                )]
        driver.close()
        unique_nodes = {str(dict(node).get("id", "")): dict(node) for node in nodes if node}
        serial_nodes = list(unique_nodes.values())
        scope = "global" if database == os.getenv("NEO4J_GLOBAL_DATABASE", "neo4j") else "interactive"
        return {"metadata": {"source": "live", "read_only": scope == "global", "scope": scope},
                "graph": {"nodes": serial_nodes, "links": links},
                "stats": {"num_nodes": len(serial_nodes), "num_edges": len(links), "num_triples": len(links)}}
    except Exception:
        return None


@app.get("/api/graph/global")
def global_graph(q: str = Query(default="", max_length=120), limit: int = Query(default=150, ge=1, le=500)) -> dict[str, Any]:
    return _neo4j_graph(database=os.getenv("NEO4J_GLOBAL_DATABASE", "neo4j"), user_env="NEO4J_GLOBAL_USER",
                        password_env="NEO4J_GLOBAL_PASSWORD", query=q, limit=limit) or _sample_global_graph()


@app.get("/api/graphs/global")
def graphs_global(q: str = Query(default="", max_length=120), limit: int = Query(default=150, ge=1, le=500)) -> dict[str, Any]:
    return global_graph(q=q, limit=limit)


@app.get("/api/graph")
def graph_read() -> dict[str, Any]:
    return global_graph()


@app.get("/api/graph/{node_id}")
def graph_node(node_id: str) -> dict[str, Any]:
    return _neo4j_graph(database=os.getenv("NEO4J_GLOBAL_DATABASE", "neo4j"), user_env="NEO4J_GLOBAL_USER",
                        password_env="NEO4J_GLOBAL_PASSWORD", node_id=node_id) or {
                            "graph": {"nodes": [], "links": []}, "metadata": {"read_only": True, "scope": "global"}}


@app.get("/api/graphs/interactive")
def graphs_interactive(q: str = Query(default="", max_length=120), limit: int = Query(default=150, ge=1, le=500)) -> dict[str, Any]:
    return _neo4j_graph(database=os.getenv("NEO4J_INTERACTIVE_DATABASE", "interactive"), user_env="NEO4J_INTERACTIVE_USER",
                        password_env="NEO4J_INTERACTIVE_PASSWORD", query=q, limit=limit) or {
                            "metadata": {"source": "empty", "read_only": False, "scope": "interactive"},
                            "graph": {"nodes": [], "links": []}, "stats": {"num_nodes": 0, "num_edges": 0, "num_triples": 0}}


# Keep legacy Html/ untouched. The new presentation is the service homepage.
app.mount("/", StaticFiles(directory=DEMO_DIR, html=True), name="demo")
