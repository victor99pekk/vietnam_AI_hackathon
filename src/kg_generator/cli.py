"""CLI entry point for the Knowledge Graph Generator."""

from pathlib import Path
import re

import click
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv()

from kg_generator.config import GraphBackend, Language, PipelineConfig, load_config
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
    default=None,
    type=click.Choice(["en", "vi"]),
    help="Language of the input data; overrides the config file when supplied.",
)
@click.option(
    "--llm/--no-llm",
    default=None,
    help="Enable GraphGen-style joint entity/relation extraction with DeepSeek.",
)
@click.option(
    "--backend",
    "backend",
    default="networkx",
    type=click.Choice(["networkx", "neo4j"]),
    help="Graph backend: networkx (in-memory) or neo4j (on-disk, incremental-capable).",
)
@click.option(
    "--clear",
    is_flag=True,
    help="(neo4j backend only) Delete all existing nodes/relationships before building.",
)
def run(
    input_paths: tuple[str, ...],
    config_path: str | None,
    output_dir: str,
    language: str | None,
    llm: bool | None,
    backend: str,
    clear: bool,
) -> None:
    """Run the full knowledge graph generation pipeline."""
    import os

    from kg_generator.pipeline import Pipeline

    config = load_config(Path(config_path) if config_path else None)

    if input_paths:
        config.input_paths = [Path(p) for p in input_paths]
    if language is not None:
        config.language = Language(language)
    config.graph_backend = GraphBackend(backend)
    if llm is not None:
        config.use_llm = llm

    pipeline = Pipeline(config, Path(output_dir))

    if config.graph_backend == GraphBackend.NEO4J:
        # ── Neo4j-backed path: direct-to-database ──
        try:
            from neo4j import GraphDatabase
        except ImportError:
            click.echo("Neo4j driver not installed. Run: uv pip install -e \".[neo4j]\"")
            raise click.Abort()

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            metrics = pipeline.execute_neo4j(session, clear=clear)
        driver.close()

        click.echo(f"\n Done! Graph built in Neo4j ({metrics.get('num_nodes', 0)} nodes, {metrics.get('num_edges', 0)} edges)")
    else:
        # ── networkx path: in-memory build → export files ──
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
@click.option(
    "-l", "--language",
    default="en",
    show_default=True,
    type=click.Choice(["en", "vi"]),
    help="Language of the input data.",
)
@click.option(
    "--llm/--no-llm",
    default=False,
    show_default=True,
    help="Enable GraphGen-style joint entity/relation extraction with DeepSeek.",
)
def quick(
    input_paths: tuple[str, ...],
    output_dir: str,
    language: str,
    llm: bool,
) -> None:
    """Run with sensible defaults — no config file needed."""
    from kg_generator.pipeline import Pipeline

    config = PipelineConfig(
        input_paths=[Path(p) for p in input_paths],
        language=Language(language),
        use_llm=llm,
    )
    try:
        pipeline = Pipeline(config, Path(output_dir))
        pipeline.execute()
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
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


@main.command("scrape")
@click.option(
    "--seed-file",
    type=click.Path(exists=True),
    help="Text file with URLs/domains/path-prefixes (one per line).",
)
@click.option(
    "--seed-urls", "-u",
    multiple=True,
    help="Inline URLs/domains/path-prefixes (repeatable).",
)
@click.option(
    "--discovery",
    type=click.Choice(["exact", "sitemap", "crawl", "auto"]),
    default="auto",
    show_default=True,
    help="URL discovery mode.",
)
@click.option(
    "--path-prefix",
    default=None,
    help="Only collect URLs starting with this prefix.",
)
@click.option(
    "--max-pages", "-n",
    default=50,
    show_default=True,
    type=int,
    help="Maximum pages to scrape.",
)
@click.option(
    "--max-time", "-t",
    default=600,
    show_default=True,
    type=int,
    help="Maximum wall-clock seconds (0 = no limit).",
)
@click.option(
    "--depth", "-d",
    default=1,
    show_default=True,
    type=int,
    help="Crawl link depth (0 = seed pages only).",
)
@click.option(
    "--delay",
    default=2.0,
    show_default=True,
    type=float,
    help="Delay between requests in seconds.",
)
@click.option(
    "--output", "-o",
    "output_dir",
    default="./data/scraped/vn_web_default",
    type=click.Path(),
    show_default=True,
    help="Output directory for JSONL + manifest + audit.",
)
@click.option(
    "--language", "-l",
    default="vi",
    type=click.Choice(["en", "vi"]),
    show_default=True,
    help="Content language.",
)
@click.option("--dataset-name", default="vietnamese-web-corpus", help="Dataset name for manifest.")
@click.option("--version", default="v1", help="Dataset version.")
@click.option("--license", "license_str", default="Public / Legally Published Exception", help="License string.")
@click.option("--contact-info", default="info@yourdomain.vn", help="Contact for User-Agent.")
@click.option("--bot-name", default="VN-LLM-Data-Collector/1.0", help="Bot name for User-Agent.")
def scrape(
    seed_file: str | None,
    seed_urls: tuple[str, ...],
    discovery: str,
    path_prefix: str | None,
    max_pages: int,
    max_time: int,
    depth: int,
    delay: float,
    output_dir: str,
    language: str,
    dataset_name: str,
    version: str,
    license_str: str,
    contact_info: str,
    bot_name: str,
) -> None:
    """Scrape Vietnamese web sources for LLM dataset collection.

    Discovers pages via sitemap or controlled crawl, respecting robots.txt
    and rate limits. Outputs pipeline-ready JSONL with provenance audit trail.
    """
    import subprocess
    import sys
    from pathlib import Path

    output = Path(output_dir)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent.parent / "data" / "download_data" / "scraper.py"),
        "--output", str(output),
        "--discovery", discovery,
        "--max-pages", str(max_pages),
        "--max-time", str(max_time),
        "--depth", str(depth),
        "--delay", str(delay),
        "--language", language,
        "--dataset-name", dataset_name,
        "--version", version,
        "--license", license_str,
        "--contact-info", contact_info,
        "--bot-name", bot_name,
    ]
    if seed_file:
        cmd.extend(["--seed-file", seed_file])
    elif seed_urls:
        cmd.extend(["--seed-urls", *seed_urls])
    else:
        raise click.UsageError("Either --seed-file or --seed-urls is required.")
    if path_prefix:
        cmd.extend(["--path-prefix", path_prefix])

    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Scraper failed with exit code {e.returncode}") from e


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
            click.echo("Replacing matching documents and preserving unrelated graph data...")

        # Upload nodes — separate by type (already normalized by exporter)
        nodes = data["graph"]["nodes"]
        chunks = [n for n in nodes if n.get("type") == "Chunk"]
        documents = [n for n in nodes if n.get("type") == "Document"]
        entities = [n for n in nodes if n.get("type") not in ("Chunk", "Document")]

        if not clear:
            from kg_generator.export.neo4j_upload import replace_documents

            replace_documents(session, [document.get("id", "") for document in documents])

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
                    "r.evidenceSentences = reduce(items = coalesce(r.evidenceSentences, []), "
                    "item IN $evidence_sentences | "
                    "CASE WHEN item IN items THEN items ELSE items + [item] END), "
                    "r.sourceChunkIds = reduce(ids = coalesce(r.sourceChunkIds, []), "
                    "chunk_id IN $source_chunk_ids | "
                    "CASE WHEN chunk_id IN ids THEN ids ELSE ids + [chunk_id] END), "
                    "r.description = CASE WHEN $description <> '' "
                    "THEN $description ELSE coalesce(r.description, '') END",
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


@main.command("add-doc")
@click.option(
    "-i", "--input",
    "input_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Input file(s) to incrementally add to the Neo4j knowledge graph.",
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
    help="Directory for output artifacts (metrics only).",
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
def add_doc(
    input_paths: tuple[str, ...],
    config_path: str | None,
    output_dir: str,
    language: str,
    llm: bool | None,
    uri: str | None,
    user: str | None,
    password: str | None,
) -> None:
    """Incrementally add one or more documents to an existing Neo4j knowledge graph.

    Only the new documents are processed — the existing graph is NOT rebuilt.
    Entities from the new documents are resolved against existing Entity nodes
    in Neo4j so that the same real-world entity is not duplicated.

    Requires a running Neo4j instance.
    """
    import os

    from kg_generator.pipeline import Pipeline

    try:
        from neo4j import GraphDatabase
    except ImportError:
        click.echo("Neo4j driver not installed. Run: uv pip install -e \".[neo4j]\"")
        raise click.Abort()

    config = load_config(Path(config_path) if config_path else None)
    config.input_paths = [Path(p) for p in input_paths]
    config.language = Language(language)
    config.graph_backend = GraphBackend.NEO4J
    if llm is not None:
        config.use_llm = llm

    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    pipeline = Pipeline(config, Path(output_dir))

    with driver.session() as session:
        metrics = pipeline.execute_neo4j(session, clear=False)

    driver.close()

    click.echo(
        f"\n Done! Added {len(input_paths)} file(s) to Neo4j "
        f"({metrics.get('num_nodes', 0)} total nodes, "
        f"{metrics.get('num_edges', 0)} total edges)"
    )


@main.command("neo4j-clear")
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
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt.",
)
def neo4j_clear(
    uri: str | None,
    user: str | None,
    password: str | None,
    yes: bool,
) -> None:
    """Delete all nodes and relationships from Neo4j."""
    import os

    try:
        from neo4j import GraphDatabase
    except ImportError:
        click.echo("Neo4j driver not installed. Run: uv pip install -e \".[neo4j]\"")
        raise click.Abort()

    if not yes:
        click.confirm(
            "This will DELETE all nodes and relationships in Neo4j. Continue?",
            abort=True,
        )

    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()

    click.echo("Neo4j database cleared.")


if __name__ == "__main__":
    main()
