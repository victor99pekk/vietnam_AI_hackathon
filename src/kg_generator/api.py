"""Small HTTP API used by the demo frontend.

The API deliberately keeps the browser contract small.  Runs are isolated in a
temporary directory and are exported as JSON by the existing pipeline.
"""

from __future__ import annotations

import re
import tempfile
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.staticfiles import StaticFiles

from kg_generator.config import Language, PipelineConfig
from kg_generator.identity import entity_id

MAX_INPUT_CHARS = 20_000

DEMO_TEXT = (
    "Võ Nguyên Giáp sinh năm 1911 tại Quảng Bình. Ông là một nhân vật quan trọng "
    "trong lịch sử Việt Nam và tham gia chỉ huy chiến dịch Điện Biên Phủ năm 1954. "
    "Chiến dịch Điện Biên Phủ góp phần kết thúc Chiến tranh Đông Dương."
)

app = FastAPI(title="AI Việt Knowledge Graph Demo", version="0.1.0")
DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"


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


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo/sample")
def demo_sample() -> dict[str, str]:
    return {"text": DEMO_TEXT, "language": "vi", "extraction": "offline"}


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
        try:
            Pipeline(config, root / "output").execute()
            output = root / "output" / "knowledge_graph.json"
            import json
            payload = json.loads(output.read_text(encoding="utf-8"))
            metrics_path = root / "output" / "metrics.json"
            payload["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            return payload
        except (RuntimeError, ImportError):
            return _fallback_graph(request.text, request.language)


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRequest) -> dict[str, Any]:
    try:
        return _run_pipeline(request)
    except Exception as error:  # stable public error; details remain server-side
        raise HTTPException(status_code=500, detail="pipeline run failed") from error


# Keep legacy Html/ untouched. The new presentation is the service homepage.
app.mount("/", StaticFiles(directory=DEMO_DIR, html=True), name="demo")
