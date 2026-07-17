"""CLI entry point for the Knowledge Graph Generator."""

from pathlib import Path

import click
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv()

from kg_generator.config import PipelineConfig, load_config


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
    from kg_generator.pipeline import Pipeline

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
    from kg_generator.pipeline import Pipeline

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
    type=click.Path(exists=True, path_type=Path),
    help="Input files or directories. All sources are deduplicated together.",
)
@click.option(
    "-m", "--source-manifest",
    "source_manifest_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML or JSON dataset provenance manifest (name, version, license, source).",
)
@click.option(
    "-o", "--output",
    "output_root",
    default="./output/curated_datasets",
    type=click.Path(path_type=Path),
    help="Root directory. Results are written to <root>/<dataset>/<version>.",
)
@click.option("--dedup-threshold", default=0.85, show_default=True, type=click.FloatRange(0, 1))
@click.option("--min-chars", default=50, show_default=True, type=click.IntRange(0, None))
@click.option("--min-words", default=10, show_default=True, type=click.IntRange(0, None))
def curate(
    input_paths: tuple[Path, ...],
    source_manifest_path: Path,
    output_root: Path,
    dedup_threshold: float,
    min_chars: int,
    min_words: int,
) -> None:
    """Create an auditable, deduplicated text dataset and quality report."""
    from kg_generator.curate.pipeline import CurationConfig, DatasetCurationPipeline

    config = CurationConfig(
        input_paths=input_paths,
        output_root=output_root,
        source_manifest_path=source_manifest_path,
        dedup_threshold=dedup_threshold,
        min_chars=min_chars,
        min_words=min_words,
    )
    try:
        output_dir = DatasetCurationPipeline(config).execute()
    except (FileExistsError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"\n Curated dataset written to {output_dir}")


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


@main.command()
@click.option(
    "-o", "--output",
    "output_dir",
    default="./output",
    type=click.Path(),
    help="Pipeline output directory containing knowledge_graph.json.",
)
def neo4j_upload(output_dir: str) -> None:
    """Clear Neo4j database and upload the generated knowledge graph."""
    from kg_generator.export.neo4j_upload import upload_from_output

    click.echo(" Clearing Neo4j database and uploading knowledge graph...")
    upload_from_output(Path(output_dir))
    click.echo(" Done! Knowledge graph uploaded to Neo4j.")


if __name__ == "__main__":
    main()
