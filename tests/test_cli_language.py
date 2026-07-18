"""Language precedence and quick-command tests."""

from click.testing import CliRunner

from kg_generator import cli
from kg_generator.config import Language


class _FakePipeline:
    instances = []

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.__class__.instances.append(self)

    def execute(self):
        return None


def test_run_preserves_yaml_language_unless_flag_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr("kg_generator.pipeline.Pipeline", _FakePipeline)
    _FakePipeline.instances.clear()
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "pipeline:\n  language: vi\n  chunk_size: 2000\n  chunk_overlap: 200\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(cli.main, ["run", "-c", str(config), "-o", str(tmp_path / "vi")])
    assert result.exit_code == 0, result.output
    assert _FakePipeline.instances[-1].config.language == Language.VIETNAMESE
    assert _FakePipeline.instances[-1].config.chunk_size == 2000
    assert _FakePipeline.instances[-1].config.chunk_overlap == 200

    result = runner.invoke(
        cli.main,
        ["run", "-c", str(config), "-l", "en", "-o", str(tmp_path / "en")],
    )
    assert result.exit_code == 0, result.output
    assert _FakePipeline.instances[-1].config.language == Language.ENGLISH


def test_quick_accepts_vietnamese_and_graphgen(monkeypatch, tmp_path):
    monkeypatch.setattr("kg_generator.pipeline.Pipeline", _FakePipeline)
    _FakePipeline.instances.clear()
    source = tmp_path / "source.txt"
    source.write_text("Dữ liệu tiếng Việt đủ dài cho lệnh kiểm thử nhanh.", encoding="utf-8")

    result = CliRunner().invoke(
        cli.main,
        ["quick", "-i", str(source), "-l", "vi", "--llm", "-o", str(tmp_path / "out")],
    )

    assert result.exit_code == 0, result.output
    config = _FakePipeline.instances[-1].config
    assert config.language == Language.VIETNAMESE
    assert config.use_llm is True
