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


def upload_graph(json_path: str | Path, clear: bool = False) -> None:
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
        if clear:
            logger.info("Clearing existing Neo4j data...")
            session.run("MATCH (n) DETACH DELETE n")

        # Create nodes
        logger.info(f"Uploading {len(nodes)} nodes...")
        for node in nodes:
            node_type = node.get("type", "Entity")
            node_id = node.get("id", "")

            is_chunk = node_type == "Chunk"
            is_document = node_type == "Document"

            if is_chunk:
                source_list = node.get("source", [])
                source_str = source_list[0] if isinstance(source_list, list) and source_list else source_list

                session.run(
                    """
                    MERGE (n:Chunk {id: $id})
                    SET n.source = $source,
                        n.text = $text,
                        n.tokenCount = $tokenCount,
                        n.index = $index,
                        n.type = $type
                    """,
                    id=node_id,
                    source=source_str,
                    text=node.get("text", ""),
                    tokenCount=node.get("tokenCount", 0),
                    index=node.get("index", 0),
                    type=node_type,
                )
            elif is_document:
                session.run(
                    """
                    MERGE (n:Document {id: $id})
                    SET n.name = $name,
                        n.type = $type,
                        n.entityType = $entity_type,
                        n.description = $description,
                        n.source = $source,
                        n.chunk_count = $chunk_count
                    """,
                    id=node_id,
                    name=node.get("name", ""),
                    type=node_type,
                    entity_type=node.get("entityType", "Document"),
                    description=node.get("description", ""),
                    source=node.get("source", []),
                    chunk_count=node.get("chunk_count", 0),
                )
            else:
                session.run(
                    """
                    MERGE (n:Entity {id: $id})
                    SET n.name = $name,
                        n.type = $type,
                        n.entityType = $entity_type,
                        n.description = $description,
                        n.importanceScore = $importance_score,
                        n.confidenceScore = $confidence_score,
                        n.embedding = $embedding
                    """,
                    id=node_id,
                    name=node.get("name", ""),
                    type=node_type,
                    entity_type=node.get("entityType", node_type),
                    description=node.get("description", ""),
                    importance_score=node.get("importanceScore", 0.0),
                    confidence_score=node.get("confidenceScore", 1.0),
                    embedding=node.get("embedding"),
                )

        # Create relationships
        logger.info(f"Uploading {len(edges)} relationships...")
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            predicates = edge.get("predicates", ["related_to"])

            for pred in predicates:
                relation_records = [
                    relation for relation in edge.get("relations", [])
                    if relation.get("predicate") == pred
                ]
                descriptions = list(dict.fromkeys(
                    relation.get("description", "")
                    for relation in relation_records
                    if relation.get("description")
                ))
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

                if pred in {"NEXT", "PART_OF", "MENTIONS"}:
                    merge_clause = f"MERGE (a)-[r:{pred}]->(b)"
                    predicate_value = None
                elif pred == "RELATION":
                    merge_clause = "MERGE (a)-[r:RELATION]->(b)"
                    predicate_value = None
                else:
                    merge_clause = "MERGE (a)-[r:RELATION {predicate: $pred}]->(b)"
                    predicate_value = pred

                session.run(
                    f"""
                    MATCH (a {{id: $source}}), (b {{id: $target}})
                    {merge_clause}
                    SET r.evidenceSentences = $evidence_sentences,
                        r.sourceChunkIds = $source_chunk_ids,
                        r.description = $description
                    """,
                    source=source,
                    target=target,
                    pred=predicate_value,
                    evidence_sentences=evidence_sentences,
                    source_chunk_ids=source_chunk_ids,
                    description=descriptions[0] if descriptions else "",
                )

    driver.close()
    logger.info(f"Successfully uploaded {len(nodes)} nodes and {len(edges)} edges to Neo4j")


def upload_from_output(output_dir: str | Path, clear: bool = False) -> None:
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

    upload_graph(json_path, clear=clear)
