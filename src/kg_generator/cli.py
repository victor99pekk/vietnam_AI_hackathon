"""CLI entry point for the Knowledge Graph Generator."""

from pathlib import Path
import re

import click
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv()

from kg_generator.config import Language, PipelineConfig, load_config
from kg_generator.curate.pipeline import CurationConfig, DatasetCurationPipeline


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
    default=None,
    help="Enable GraphGen-style joint entity/relation extraction with DeepSeek.",
)
def run(
    input_paths: tuple[str, ...],
    config_path: str | None,
    output_dir: str,
    language: str,
    llm: bool | None,
) -> None:
    """Run the full knowledge graph generation pipeline."""
    from kg_generator.pipeline import Pipeline

    config = load_config(Path(config_path) if config_path else None)

    if input_paths:
        config.input_paths = [Path(p) for p in input_paths]
    config.language = Language(language)
    if llm is not None:
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


@main.command("curate")
@click.option(
    "-i", "--input", "input_paths", multiple=True, required=True,
    type=click.Path(exists=True), help="Input file or directory paths (repeatable).",
)
@click.option(
    "-m", "--manifest", "manifest_path", required=True,
    type=click.Path(exists=True), help="YAML or JSON source-provenance manifest.",
)
@click.option(
    "-o", "--output", "output_root", default="./output/curated_datasets",
    type=click.Path(), show_default=True, help="Root for immutable curated dataset versions.",
)
@click.option("--surface-threshold", default=0.90, show_default=True, type=click.FloatRange(0, 1))
@click.option("--semantic-review-threshold", default=0.92, show_default=True, type=click.FloatRange(0, 1))
@click.option("--semantic-model", default="BAAI/bge-m3", show_default=True)
@click.option("--semantic-model-revision", default=None, help="Pinned Hugging Face model revision.")
@click.option("--semantic-review/--no-semantic-review", default=True, show_default=True)
@click.option("--device", default="cuda", show_default=True, help="Embedding device, normally cuda or cpu.")
@click.option("--max-record-tokens", default=2048, show_default=True, type=click.IntRange(3, None))
@click.option("--embedding-batch-tokens", default=8192, show_default=True, type=click.IntRange(3, None))
@click.option("--shard-tokens", default=1_000_000, show_default=True, type=click.IntRange(3, None))
@click.option("--resume", is_flag=True, help="Resume a matching incomplete staging version.")
def curate(
    input_paths: tuple[str, ...],
    manifest_path: str,
    output_root: str,
    surface_threshold: float,
    semantic_review_threshold: float,
    semantic_model: str,
    semantic_model_revision: str | None,
    semantic_review: bool,
    device: str,
    max_record_tokens: int,
    embedding_batch_tokens: int,
    shard_tokens: int,
    resume: bool,
) -> None:
    """Create an immutable, audited LLM-ready text dataset."""
    config = CurationConfig(
        input_paths=tuple(Path(path) for path in input_paths),
        output_root=Path(output_root),
        source_manifest_path=Path(manifest_path),
        surface_dedup_threshold=surface_threshold,
        semantic_review_threshold=semantic_review_threshold,
        semantic_model=semantic_model,
        semantic_model_revision=semantic_model_revision,
        semantic_review_enabled=semantic_review,
        device=device,
        max_record_tokens=max_record_tokens,
        embedding_batch_token_budget=embedding_batch_tokens,
        shard_token_budget=shard_tokens,
        resume=resume,
    )
    try:
        output_dir = DatasetCurationPipeline(config).execute()
    except (FileExistsError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Curated dataset written to {output_dir}")


@main.command()
@click.option(
    "-o", "--output",
    "output_dir",
    required=True,
    type=click.Path(exists=True),
    help="Output directory containing knowledge_graph.json from a previous run.",
)
@click.option(
    "--uri",
    default=None,
    help="Neo4j connection URI (default: $NEO4J_URI or bolt://localhost:7687).",
)
@click.option(
    "--user",
    default=None,
    help="Neo4j username (default: $NEO4J_USER or neo4j).",
)
@click.option(
    "--password",
    default=None,
    help="Neo4j password (default: $NEO4J_PASSWORD).",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Delete all existing Neo4j nodes and relationships before uploading.",
)
def neo4j_upload(
    output_dir: str,
    uri: str | None,
    user: str | None,
    password: str | None,
    clear: bool,
) -> None:
    """Upload a generated knowledge graph to a Neo4j database."""
    import json
    import os

    output_path = Path(output_dir)
    kg_file = output_path / "knowledge_graph.json"

    if not kg_file.exists():
        raise click.FileError(str(kg_file), hint="Run the pipeline first: kg-gen run -i data/ -o output")

    with open(kg_file) as f:
        data = json.load(f)

    # Resolve connection details
    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        click.echo("Neo4j driver not installed. Run: uv pip install -e \".[neo4j]\"")
        raise click.Abort()

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        if clear:
            click.echo("Clearing existing graph (--clear was supplied)...")
            session.run("MATCH (n) DETACH DELETE n")
        else:
            click.echo("Preserving existing graph (use --clear for a clean replacement)...")

        # Upload nodes — separate by type (already normalized by exporter)
        nodes = data["graph"]["nodes"]
        chunks = [n for n in nodes if n.get("type") == "Chunk"]
        documents = [n for n in nodes if n.get("type") == "Document"]
        entities = [n for n in nodes if n.get("type") not in ("Chunk", "Document")]

        # Upload Document nodes
        click.echo(f"Uploading {len(documents)} Document nodes...")
        for doc in documents:
            session.run(
                "MERGE (d:Document {id: $id}) "
                "SET d.name = $name, "
                "d.type = $type, "
                "d.description = $description, "
                "d.source = $source, "
                "d.chunk_count = $chunk_count "
                "REMOVE d.entityType",
                name=doc["name"],
                id=doc["id"],
                type=doc.get("type", "Document"),
                description=doc.get("description", ""),
                source=doc.get("source", []),
                chunk_count=doc.get("chunk_count", 0),
            )

        # Upload Entity nodes with clean GraphRAG properties
        click.echo(f"Uploading {len(entities)} Entity nodes...")
        for node in entities:
            session.run(
                "MERGE (n:Entity {id: $id}) "
                "SET n.name = $name, "
                "n.type = $type, "
                "n.description = $description, "
                "n.importanceScore = $importanceScore, "
                "n.confidenceScore = $confidenceScore, "
                "n.embedding = $embedding "
                "REMOVE n.entityType",
                name=node["name"],
                id=node["id"],
                type=node.get("type", "Entity"),
                description=node.get("description", ""),
                importanceScore=node.get("importanceScore", 0.0),
                confidenceScore=node.get("confidenceScore", 1.0),
                embedding=node.get("embedding"),
            )

        # Upload Chunk nodes with deterministic IDs from the pipeline
        click.echo(f"Uploading {len(chunks)} Chunk nodes...")
        for chunk in chunks:
            session.run(
                "MERGE (c:Chunk {id: $id}) "
                "SET c.type = $type, "
                "c.source = $source, "
                "c.text = $text, "
                "c.tokenCount = $tokenCount, "
                "c.index = $index "
                "REMOVE c.entityType",
                id=chunk["id"],
                type=chunk.get("type", "Chunk"),
                source=chunk.get("source", ""),
                text=chunk.get("text", ""),
                tokenCount=chunk.get("tokenCount", 0),
                index=chunk.get("index", 0),
            )

        # Upload edges using stable IDs for every node type
        edges = data["graph"]["links"] if "links" in data["graph"] else data["graph"]["edges"]
        click.echo(f"Uploading {len(edges)} relationships...")
        edge_count = 0
        for edge in edges:
            predicates = edge.get("predicates", ["RELATION"])
            for pred in predicates:
                safe_pred = re.sub(r"[^A-Z0-9_]", "_", pred.upper()).strip("_")
                safe_pred = safe_pred or "RELATION"
                relation_records = [
                    relation
                    for relation in edge.get("relations", [])
                    if relation.get("predicate") == pred
                ]
                evidence_sentences = list(dict.fromkeys(
                    relation.get("evidence_sentence", "")
                    for relation in relation_records
                    if relation.get("evidence_sentence")
                ))
                source_chunk_ids = list(dict.fromkeys(
                    relation.get("source_chunk_id", "")
                    for relation in relation_records
                    if relation.get("source_chunk_id")
                ))
                descriptions = list(dict.fromkeys(
                    relation.get("description", "")
                    for relation in relation_records
                    if relation.get("description")
                ))
                session.run(
                    "MATCH (a {id: $source}) "
                    "MATCH (b {id: $target}) "
                    f"MERGE (a)-[r:{safe_pred}]->(b) "
                    "SET r.weight = $weight, "
                    "r.evidenceSentences = $evidence_sentences, "
                    "r.sourceChunkIds = $source_chunk_ids, "
                    "r.description = $description",
                    source=edge["source"],
                    target=edge["target"],
                    weight=edge.get("weight", 1),
                    evidence_sentences=evidence_sentences,
                    source_chunk_ids=source_chunk_ids,
                    description=descriptions[0] if descriptions else "",
                )
                edge_count += 1

        click.echo(f"Done! Uploaded {len(documents)} docs, {len(chunks)} chunks, {len(entities)} entities, {edge_count} relationships to {uri}")

    driver.close()


@main.command("neo4j-download")
@click.option(
    "-o", "--output",
    "output_path",
    default="./generated_KGs/output/knowledge_graph.json",
    type=click.Path(),
    help="Path to save the downloaded knowledge_graph.json "
         "(default: ./generated_KGs/output/knowledge_graph.json).",
)
@click.option(
    "--uri",
    default=None,
    help="Neo4j connection URI (default: $NEO4J_URI or bolt://localhost:7687).",
)
@click.option(
    "--user",
    default=None,
    help="Neo4j username (default: $NEO4J_USER or neo4j).",
)
@click.option(
    "--password",
    default=None,
    help="Neo4j password (default: $NEO4J_PASSWORD).",
)
def neo4j_download(
    output_path: str,
    uri: str | None,
    user: str | None,
    password: str | None,
) -> None:
    """Download a knowledge graph from Neo4j and save as knowledge_graph.json.

    Connects to Neo4j using credentials from .env (or CLI overrides),
    downloads all nodes and relationships, and reconstructs the
    knowledge_graph.json format expected by the evaluation pipeline.

    Typical workflow:
      1. Local:  kg-gen run ... && kg-gen neo4j-upload -o output/
      2. Colab:  kg-gen neo4j-download -o generated_KGs/output/
      3. Colab:  python evaluation/run_eval.py --method all --kg generated_KGs/output/knowledge_graph.json
    """
    import os

    # Override env vars with CLI options if provided
    if uri:
        os.environ["NEO4J_URI"] = uri
    if user:
        os.environ["NEO4J_USER"] = user
    if password:
        os.environ["NEO4J_PASSWORD"] = password

    try:
        from neo4j import GraphDatabase  # noqa: F401 — validate import
    except ImportError:
        click.echo("Neo4j driver not installed. Run: uv pip install -e \".[neo4j]\"")
        raise click.Abort()

    from kg_generator.export.neo4j_upload import download_graph

    download_graph(output_path)
    click.echo(f"\n Done! Graph saved to {output_path}")


if __name__ == "__main__":
    main()
