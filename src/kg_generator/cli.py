"""CLI entry point for the Knowledge Graph Generator."""

from pathlib import Path

import click
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv()

from kg_generator.config import Language, PipelineConfig, load_config
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
    config.language = Language(language)
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
def neo4j_upload(output_dir: str, uri: str | None, user: str | None, password: str | None) -> None:
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

    import hashlib
    import secrets

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Wipe existing graph to avoid stale labels/properties from previous uploads
        click.echo("Clearing existing graph...")
        session.run("MATCH (n) DETACH DELETE n")

        # Upload nodes — separate by type (already normalized by exporter)
        nodes = data["graph"]["nodes"]
        chunks = [n for n in nodes if n.get("type") == "Chunk"]
        documents = [n for n in nodes if n.get("type") == "Document"]
        entities = [n for n in nodes if n.get("type") not in ("Chunk", "Document")]

        # Upload Document nodes
        click.echo(f"Uploading {len(documents)} Document nodes...")
        for doc in documents:
            session.run(
                "MERGE (d:Document {name: $name}) "
                "SET d.id = $id, "
                "d.entityType = $entityType, "
                "d.description = $description, "
                "d.source = $source, "
                "d.chunk_count = $chunk_count",
                name=doc["name"],
                id=doc["id"],
                entityType=doc.get("entityType", "Document"),
                description=doc.get("description", ""),
                source=doc.get("source", []),
                chunk_count=doc.get("chunk_count", 0),
            )

        # Upload Entity nodes with clean GraphRAG properties
        click.echo(f"Uploading {len(entities)} Entity nodes...")
        for node in entities:
            session.run(
                "MERGE (n:Entity {name: $name}) "
                "SET n.id = $id, "
                "n.entityType = $entityType, "
                "n.description = $description, "
                "n.importanceScore = $importanceScore, "
                "n.confidenceScore = $confidenceScore, "
                "n.embedding = $embedding",
                name=node["name"],
                id=node["id"],
                entityType=node.get("entityType", "Entity"),
                description=node.get("description", ""),
                importanceScore=node.get("importanceScore", 0.0),
                confidenceScore=node.get("confidenceScore", 1.0),
                embedding=node.get("embedding"),
            )

        # Upload Chunk nodes with random hash id
        click.echo(f"Uploading {len(chunks)} Chunk nodes...")
        for chunk in chunks:
            raw = secrets.token_bytes(32)
            chunk_id = hashlib.sha256(raw).hexdigest()[:16]

            session.run(
                "CREATE (c:Chunk {id: $id}) "
                "SET c.source = $source, "
                "c.text = $text, "
                "c.tokenCount = $tokenCount, "
                "c.index = $index",
                id=chunk_id,
                source=chunk.get("source", ""),
                text=chunk.get("text", ""),
                tokenCount=chunk.get("tokenCount", 0),
                index=chunk.get("index", 0),
            )

        # Upload edges — match on any label using name property
        edges = data["graph"]["links"] if "links" in data["graph"] else data["graph"]["edges"]
        click.echo(f"Uploading {len(edges)} relationships...")
        edge_count = 0
        for edge in edges:
            predicates = edge.get("predicates", ["RELATED_TO"])
            for pred in predicates:
                safe_pred = pred.upper().replace(" ", "_")
                session.run(
                    "MATCH (a {name: $source}) "
                    "MATCH (b {name: $target}) "
                    f"MERGE (a)-[r:{safe_pred}]->(b) "
                    "SET r.weight = $weight",
                    source=edge["source"],
                    target=edge["target"],
                    weight=edge.get("weight", 1),
                )
                edge_count += 1

        # Strip name property from Entity nodes (no longer needed after edges are created)
        click.echo("Removing name property from Entity nodes...")
        session.run("MATCH (n:Entity) REMOVE n.name")

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
