"""Upload a generated knowledge graph to a Neo4j database."""

import json
import logging
import os
from pathlib import Path

from kg_generator.config import load_config

logger = logging.getLogger(__name__)


def _get_connection():
    """Create and return a Neo4j driver using env vars."""
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "")
    user = os.environ.get("NEO4J_USER", "")
    password = os.environ.get("NEO4J_PASSWORD", "")

    if not uri or not user or not password:
        raise RuntimeError(
            "Neo4j credentials not set. Ensure NEO4J_URI, NEO4J_USER, "
            "and NEO4J_PASSWORD are defined in .env"
        )

    return GraphDatabase.driver(uri, auth=(user, password))


def clear_database():
    """Delete all nodes and relationships from the Neo4j database."""
    driver = _get_connection()
    with driver.session() as session:
        summary = session.run("MATCH (n) DETACH DELETE n").consume()
        logger.info("Cleared all nodes and relationships from Neo4j")
    driver.close()


def upload_graph(json_path: str | Path) -> None:
    """Read a knowledge_graph.json and upload its nodes + edges to Neo4j."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Graph file not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    graph_data = data.get("graph", {})
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("links", [])

    driver = _get_connection()

    with driver.session() as session:
        # Clear existing data
        logger.info("Clearing existing Neo4j data...")
        session.run("MATCH (n) DETACH DELETE n")

        # Create nodes
        logger.info(f"Uploading {len(nodes)} nodes...")
        for node in nodes:
            node_id = node.get("id", "")
            label = node.get("label", "Entity")
            mentions = node.get("mentions", [])
            confidence = node.get("confidence", 1.0)

            session.run(
                """
                CREATE (n:Entity:%s {
                    id: $id,
                    name: $id,
                    label: $label,
                    confidence: $confidence,
                    mentions: $mentions
                })
                """ % label,
                id=node_id,
                label=label,
                confidence=confidence,
                mentions=mentions,
            )

        # Create relationships
        logger.info(f"Uploading {len(edges)} relationships...")
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            predicates = edge.get("predicates", ["related_to"])

            for pred in predicates:
                session.run(
                    """
                    MATCH (a {id: $source}), (b {id: $target})
                    CREATE (a)-[:RELATES_TO {predicate: $pred}]->(b)
                    """,
                    source=source,
                    target=target,
                    pred=pred,
                )

    driver.close()
    logger.info(f"Successfully uploaded {len(nodes)} nodes and {len(edges)} edges to Neo4j")


def upload_from_output(output_dir: str | Path) -> None:
    """Upload a knowledge graph from a pipeline output directory to Neo4j."""
    output_dir = Path(output_dir)
    json_path = output_dir / "knowledge_graph.json"

    if not json_path.exists():
        # Try finding in neo4j_import subfolder
        json_path = output_dir / "neo4j_import" / "knowledge_graph.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"No knowledge_graph.json found in {output_dir}. "
            f"Run the pipeline with 'json' in export_formats first."
        )

    upload_graph(json_path)
