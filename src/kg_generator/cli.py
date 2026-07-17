"""CLI entry point for the Knowledge Graph Generator."""

import sys
from pathlib import Path

import click

from kg_generator.config import PipelineConfig, load_config
from kg_generator.pipeline import Pipeline


@click.group()
@click.version_option(version="0.1.0", prog_name="kg-generator")
def main() -> None:
    """Knowledge Graph Generator — build structured KGs from raw text for LLM training & Graph RAG."""


@main.command()
@click.option(
    "-i", "--input",
    "input_paths",
    multiple=True,
    type=click.Path(exists=True),
    help="Input file or directory paths (can be repeated).",
)
@click.option(
    "-c", "--config",
    "config_path",
    type=click.Path(exists=True),
    help="Path to a YAML pipeline configuration file.",
)
@click.option(
    "-o", "--output",
    "output_dir",
    default="./output",
    type=click.Path(),
    help="Directory for output artifacts.",
)
@click.option(
    "-l", "--language",
    default="en",
    type=click.Choice(["en", "vi"]),
    help="Language of the input data.",
)
@click.option(
    "--llm/--no-llm",
    default=False,
    help="Enable LLM-based entity/relation extraction.",
)
def run(
    input_paths: tuple[str, ...],
    config_path: str | None,
    output_dir: str,
    language: str,
    llm: bool,
) -> None:
    """Run the full knowledge graph generation pipeline."""
    config = load_config(Path(config_path) if config_path else None)

    if input_paths:
        config.input_paths = [Path(p) for p in input_paths]
    config.language = language  # type: ignore[assignment]
    config.use_llm = llm

    pipeline = Pipeline(config, Path(output_dir))
    pipeline.execute()

    click.echo(f"\n Done! Output written to {output_dir}")


@main.command()
@click.option(
    "-i", "--input",
    "input_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Input file or directory paths.",
)
@click.option(
    "-o", "--output",
    "output_dir",
    default="./output",
    type=click.Path(),
    help="Directory for output.",
)
def quick(input_paths: tuple[str, ...], output_dir: str) -> None:
    """Run with sensible defaults — no config file needed."""
    config = PipelineConfig(input_paths=[Path(p) for p in input_paths])
    pipeline = Pipeline(config, Path(output_dir))
    pipeline.execute()
    click.echo(f"\n Done! Output written to {output_dir}")


@main.command()
@click.option(
    "-i", "--input",
    "input_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Input files to evaluate.",
)
@click.option(
    "-r", "--reference",
    "reference_path",
    type=click.Path(exists=True),
    help="Reference/gold-standard graph for comparison.",
)
def evaluate(input_paths: tuple[str, ...], reference_path: str | None) -> None:
    """Run only the quality evaluation stage on existing data."""
    from kg_generator.evaluate.metrics import QualityEvaluator

    evaluator = QualityEvaluator()
    for path in input_paths:
        results = evaluator.evaluate(Path(path))
        click.echo(f"\n--- {path} ---")
        for metric, score in results.items():
            click.echo(f"  {metric}: {score:.3f}")


if __name__ == "__main__":
    main()
