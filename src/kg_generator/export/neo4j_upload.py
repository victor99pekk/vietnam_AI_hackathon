"""Upload/download a knowledge graph to/from a Neo4j database."""

import hashlib
import json
import logging
import os
import secrets
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
            node_type = node.get("type", "Entity")
            node_id = node.get("id", "")
            mentions = node.get("mentions", [])
            confidence = node.get("confidence", 1.0)

            is_chunk = node_type == "Chunk"

            if is_chunk:
                # Chunk nodes: random hash id, source as attribute, no name
                raw = secrets.token_bytes(32)
                chunk_id = hashlib.sha256(raw).hexdigest()[:16]
                source_list = node.get("source", [])
                source_str = source_list[0] if source_list else ""

                session.run(
                    """
                    CREATE (n:Entity:Chunk {
                        id: $id,
                        source: $source,
                        text: $text,
                        tokenCount: $tokenCount,
                        index: $index,
                        confidence: $confidence
                    })
                    """,
                    id=chunk_id,
                    source=source_str,
                    text=node.get("text", ""),
                    tokenCount=node.get("tokenCount", 0),
                    index=node.get("index", 0),
                    confidence=confidence,
                )
            else:
                session.run(
                    """
                    CREATE (n:Entity:%s {
                        id: $id,
                        name: $id,
                        label: $label,
                        confidence: $confidence,
                        mentions: $mentions
                    })
                    """ % node_type,
                    id=node_id,
                    label=node_type,
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


def download_graph(output_path: str | Path) -> None:
    """Download the full knowledge graph from Neo4j and save as knowledge_graph.json.

    Reconstructs the format expected by the evaluation pipeline
    (evaluation/run_eval.py) from Neo4j. Reads connection details from
    .env (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).

    Use case: generate KG locally → upload to Neo4j → download on Colab for eval.
    """
    output_path = Path(output_path)
    driver = _get_connection()

    with driver.session() as session:
        # ── Fetch Entity nodes ──
        entity_result = session.run(
            "MATCH (n:Entity) "
            "RETURN n.id AS id, n.entityType AS entityType, "
            "n.description AS description, n.importanceScore AS importanceScore, "
            "n.confidenceScore AS confidenceScore, n.embedding AS embedding"
        )
        entity_nodes = [dict(record) for record in entity_result]

        # ── Fetch Document nodes ──
        doc_result = session.run(
            "MATCH (d:Document) "
            "RETURN d.id AS id, d.name AS name, d.entityType AS entityType, "
            "d.description AS description, d.source AS source, "
            "d.chunk_count AS chunk_count"
        )
        doc_nodes = [dict(record) for record in doc_result]

        # ── Fetch Chunk nodes ──
        chunk_result = session.run(
            "MATCH (c:Chunk) "
            "RETURN c.id AS id, c.source AS source, c.text AS text, "
            "c.tokenCount AS tokenCount, c.index AS index"
        )
        chunk_nodes = [dict(record) for record in chunk_result]

        # ── Fetch relationships ──
        rel_result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN a.id AS source, b.id AS target, type(r) AS predicate, "
            "r.weight AS weight"
        )
        relationships = [dict(record) for record in rel_result]

    driver.close()

    if not entity_nodes and not doc_nodes and not chunk_nodes:
        raise RuntimeError(
            "Neo4j database is empty — no nodes found. "
            "Upload a graph first: make upload"
        )

    # ── Reconstruct knowledge_graph.json ──────────────────────

    # Build graph.nodes — one node dict per Neo4j node
    graph_nodes: list[dict] = []

    for en in entity_nodes:
        node: dict = {
            "id": en["id"],
            "type": en.get("entityType", "Entity"),
            "entityType": en.get("entityType", "Entity"),
            "description": en.get("description", ""),
            "importanceScore": en.get("importanceScore", 0.0),
            "confidenceScore": en.get("confidenceScore", 1.0),
        }
        if en.get("embedding") is not None:
            node["embedding"] = en["embedding"]
        graph_nodes.append(node)

    for dn in doc_nodes:
        graph_nodes.append({
            "id": dn["id"],
            "type": "Document",
            "name": dn.get("name", dn["id"]),
            "entityType": dn.get("entityType", "Document"),
            "description": dn.get("description", ""),
            "source": dn.get("source", ""),
            "chunk_count": dn.get("chunk_count", 0),
        })

    for cn in chunk_nodes:
        graph_nodes.append({
            "id": cn["id"],
            "type": "Chunk",
            "source": cn.get("source", ""),
            "text": cn.get("text", ""),
            "tokenCount": cn.get("tokenCount", 0),
            "index": cn.get("index", 0),
        })

    # Build graph.links — group by (source, target) to collect predicates
    links_map: dict[tuple[str, str], list[str]] = {}
    for rel in relationships:
        key = (rel["source"], rel["target"])
        if key not in links_map:
            links_map[key] = []
        links_map[key].append(rel["predicate"])

    graph_links = [
        {"source": src, "target": tgt, "predicates": preds}
        for (src, tgt), preds in links_map.items()
    ]

    # Build entities list (Entity-type nodes only; used by evaluation code)
    entities = []
    for en in entity_nodes:
        entities.append({
            "name": en["id"],
            "type": en.get("entityType", "Entity"),
            "description": en.get("description", ""),
        })

    # Build triples list — deduplicate on (subj, pred, obj)
    seen_triples: set[tuple[str, str, str]] = set()
    triples: list[list[str]] = []
    for rel in relationships:
        key = (rel["source"], rel["predicate"], rel["target"])
        if key not in seen_triples:
            seen_triples.add(key)
            triples.append([rel["source"], rel["predicate"], rel["target"], ""])

    # Assemble final output
    output = {
        "graph": {
            "nodes": graph_nodes,
            "links": graph_links,
        },
        "entities": entities,
        "triples": triples,
        "stats": {
            "num_nodes": len(graph_nodes),
            "num_edges": len(graph_links),
            "num_triples": len(triples),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(
        "Downloaded %d nodes, %d edges, %d triples → %s",
        len(graph_nodes), len(graph_links), len(triples), output_path,
    )
