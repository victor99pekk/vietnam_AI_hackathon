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

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Create uniqueness constraints
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")

        # Upload nodes — separate Chunk nodes from regular Entity nodes
        nodes = data["graph"]["nodes"]
        chunks = [n for n in nodes if n.get("type") == "Chunk"]
        entities = [n for n in nodes if n.get("type") != "Chunk"]

        # Upload Entity nodes with clean GraphRAG properties
        click.echo(f"Uploading {len(entities)} Entity nodes...")
        for node in entities:
            node_type = node.get("type", "Entity")
            session.run(
                "MERGE (n:Entity {name: $name}) "
                f"SET n:`{node_type}` "
                "SET n.id = $id, "
                "n.type = $type, "
                "n.aliases = $aliases, "
                "n.description = $description, "
                "n.importanceScore = $importanceScore, "
                "n.confidenceScore = $confidenceScore, "
                "n.source = $source, "
                "n.embedding = $embedding, "
                "n.updatedAt = $updatedAt",
                name=node.get("name", node.get("id", "")),
                id=node.get("id", ""),
                type=node_type,
                aliases=node.get("aliases", []),
                description=node.get("description", ""),
                importanceScore=node.get("importanceScore", 0.0),
                confidenceScore=node.get("confidenceScore", 1.0),
                source=node.get("source", []),
                embedding=node.get("embedding") if isinstance(node.get("embedding"), list) else None,
                updatedAt=node.get("updatedAt", ""),
            )

        # Upload Chunk nodes with clean GraphRAG properties
        click.echo(f"Uploading {len(chunks)} Chunk nodes...")
        for chunk in chunks:
            chunk_id = chunk.get("name", chunk.get("id", ""))
            session.run(
                "MERGE (c:Chunk {name: $name}) "
                "SET c.id = $id, "
                "c.text = $text, "
                "c.tokenCount = $tokenCount, "
                "c.index = $index, "
                "c.embedding = $embedding, "
                "c.source = $source",
                name=chunk_id,
                id=chunk_id,
                text=chunk.get("text", ""),
                tokenCount=chunk.get("tokenCount", 0),
                index=chunk.get("index", 0),
                embedding=chunk.get("embedding") if isinstance(chunk.get("embedding"), list) else None,
                source=chunk.get("source", ""),
            )

        # Strip legacy properties left over from previous uploads
        click.echo("Cleaning up legacy properties...")
        session.run(
            "MATCH (n:Entity) "
            "REMOVE n.confidence, n.mentions, n.label, n.displayName, n.entity_id, n.attributes"
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

        click.echo(f"Done! Uploaded {len(nodes)} nodes and {edge_count} relationships to {uri}")

    driver.close()


if __name__ == "__main__":
    main()
