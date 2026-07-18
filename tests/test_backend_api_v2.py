import threading

from kg_generator.api import (
    RUN_LOCK,
    RunOptions,
    RunRequest,
    SourceRequest,
    _config_for_run,
    create_run,
    global_graph,
    options,
)


def test_options_include_semantic_deduplication():
    payload = options()
    assert "semantic" in payload["document_dedup_methods"]
    assert "semantic" in payload["dedup_methods"]


def test_nested_options_map_to_pipeline_config(tmp_path):
    options = RunOptions(language="en", extraction="graphgen", llm_model="deepseek-v4-pro", chunk_method="sentence", chunk_size=0)
    config = _config_for_run(options, tmp_path / "source.json")
    assert config.language.value == "en"
    assert config.use_llm is True
    assert config.llm_model == "deepseek-v4-pro"
    assert config.chunk_method == "sentence"


def test_global_graph_falls_back_without_neo4j(monkeypatch):
    for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    payload = global_graph()
    assert payload["metadata"]["read_only"] is True
    assert payload["graph"]["nodes"]


def test_run_lock_is_nonblocking(monkeypatch):
    monkeypatch.setenv("KG_DEMO_USE_FULL_PIPELINE", "0")
    assert RUN_LOCK.acquire(blocking=False)
    try:
        request = RunRequest(source=SourceRequest(text="Hồ Chí Minh ở Việt Nam."))
        try:
            create_run(request)
        except Exception as error:
            assert getattr(error, "status_code", None) == 409
        else:
            raise AssertionError("concurrent run should be rejected")
    finally:
        RUN_LOCK.release()
