"""Vietnamese core pipeline contract tests."""

import json
from types import SimpleNamespace

import kg_generator.pipeline as pipeline_module
from kg_generator.config import Language, PipelineConfig
from kg_generator.extract.entities import VietnameseExtractor
from kg_generator.extract.graphgen import GraphGenExtractor
from kg_generator.pipeline import Pipeline


class _FakeDeepSeekClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=next(self.responses))
                )
            ]
        )


def _source(tmp_path):
    source = tmp_path / "vietnamese.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "giap",
                "title": "Đại tướng Võ Nguyên Giáp",
                "url": "https://example.vn/vo-nguyen-giap",
                "license": "Demo license",
                "source_domain": "example.vn",
                "text": (
                    "Võ Nguyên Giáp sinh năm 1911 tại Quảng Bình và là một nhân vật "
                    "quan trọng trong lịch sử Việt Nam. Ông tham gia chỉ huy chiến dịch "
                    "Điện Biên Phủ năm 1954."
                ),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def _config(source, *, use_llm):
    return PipelineConfig(
        language=Language.VIETNAMESE,
        input_paths=[source],
        file_formats=["jsonl"],
        chunk_size=0,
        use_llm=use_llm,
        graphgen_max_gleanings=1,
        resolve_method="string",
        export_formats=["json"],
    )


def _assert_vietnamese_graph(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    node_ids = {node["id"] for node in payload["graph"]["nodes"]}
    assert payload["metadata"]["language"] == "vi"
    assert any(entity["name"] == "Võ Nguyên Giáp" for entity in payload["entities"])
    assert any(triple["predicate"] == "MENTIONS" for triple in payload["triples"])
    assert any(
        triple["predicate"] not in {"MENTIONS", "PART_OF", "NEXT"}
        for triple in payload["triples"]
    )
    assert all(
        triple["subject"] in node_ids and triple["object"] in node_ids
        for triple in payload["triples"]
    )
    semantic = {
        (triple["subject"], triple["predicate"], triple["object"])
        for triple in payload["triples"]
        if triple["predicate"] not in {"MENTIONS", "PART_OF", "NEXT"}
    }
    assert all(subject != object_ for subject, _, object_ in semantic)
    assert not any((object_, predicate, subject) in semantic for subject, predicate, object_ in semantic)
    document = next(entity for entity in payload["entities"] if entity["type"] == "Document")
    assert document["name"] == "Đại tướng Võ Nguyên Giáp"
    assert document["url"] == "https://example.vn/vo-nguyen-giap"
    assert document["source_domain"] == "example.vn"


def test_offline_vietnamese_pipeline_builds_non_empty_graph(monkeypatch, tmp_path):
    rows = [
        ("Võ", "Np", "B-NP", "B-PER"),
        ("Nguyên", "Np", "I-NP", "I-PER"),
        ("Giáp", "Np", "I-NP", "I-PER"),
        ("sinh", "V", "B-VP", "O"),
        ("Quảng Bình", "Np", "B-NP", "B-LOC"),
    ]
    monkeypatch.setattr(
        pipeline_module,
        "VietnameseExtractor",
        lambda: VietnameseExtractor(ner_function=lambda _text: rows),
    )
    output = tmp_path / "offline"

    Pipeline(_config(_source(tmp_path), use_llm=False), output).execute()

    _assert_vietnamese_graph(output / "knowledge_graph.json")


def test_graphgen_vietnamese_pipeline_does_not_require_underthesea(monkeypatch, tmp_path):
    extraction = """("entity"<|>"Võ Nguyên Giáp"<|>"person"<|>"Một nhân vật lịch sử Việt Nam.")##
("entity"<|>"Điện Biên Phủ"<|>"event"<|>"Một chiến dịch lịch sử.")##
("relationship"<|>"Võ Nguyên Giáp"<|>"Điện Biên Phủ"<|>"Ông tham gia chỉ huy chiến dịch.")<|COMPLETE|>"""
    client = _FakeDeepSeekClient([extraction, "NO"])

    def graphgen_factory(**kwargs):
        return GraphGenExtractor(client=client, **kwargs)

    monkeypatch.setattr(pipeline_module, "GraphGenExtractor", graphgen_factory)
    output = tmp_path / "graphgen"

    Pipeline(_config(_source(tmp_path), use_llm=True), output).execute()

    payload = json.loads((output / "knowledge_graph.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    _assert_vietnamese_graph(output / "knowledge_graph.json")
    assert payload["metadata"]["extraction"]["method"] == "graphgen"
    assert payload["metadata"]["extraction"]["prompt_version"].endswith("v2")
    assert payload["metadata"]["pipeline"]["chunking"]["method"] == "fixed"
    assert payload["metadata"]["pipeline"]["deduplication"]["chunk_method"] == "minhash"
    assert payload["metadata"]["processing"]["loaded_documents"] == 1
    assert 0 <= metrics["structural_audit"]["overall_health_score"] <= 100
    assert {
        "orphan_analysis",
        "density_analysis",
        "schema_compliance",
        "entity_duplication",
        "multi_hop_connectivity",
    } <= metrics["structural_audit"].keys()
