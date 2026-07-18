"""Upload/download a knowledge graph to/from a Neo4j database."""

import json
import logging
import os
from pathlib import Path

from kg_generator.config import load_config

logger = logging.getLogger(__name__)


def replace_documents(session, document_ids: list[str]) -> None:
    """Delete existing KG data owned by incoming document IDs.

    Document IDs are stable across content changes, while chunk IDs include the
    chunk text. Removing the old chunks before upload prevents changed documents
    from leaving stale chunks and extracted relationships in Neo4j.
    """
    document_ids = sorted({document_id for document_id in document_ids if document_id})
    if not document_ids:
        return

    record = session.run(
        """
        MATCH (chunk:Chunk)-[:PART_OF]->(document:Document)
        WHERE document.id IN $document_ids
        OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)
        RETURN collect(DISTINCT chunk.id) AS chunk_ids,
               collect(DISTINCT entity.id) AS entity_ids
        """,
        document_ids=document_ids,
    ).single()
    chunk_ids = list(record["chunk_ids"] or []) if record else []
    entity_ids = list(record["entity_ids"] or []) if record else []

    if chunk_ids:
        # Extracted entity-to-entity relationships are not directly connected
        # to Chunk nodes. Remove replaced chunk IDs from their provenance and
        # delete a relationship only when no other document still supports it.
        # Evidence text cannot currently be mapped to individual source chunks,
        # so clear it rather than retain text from the replaced document.
        session.run(
            """
            MATCH ()-[relationship]->()
            WHERE NOT type(relationship) IN ['PART_OF', 'NEXT', 'MENTIONS']
              AND any(chunk_id IN coalesce(relationship.sourceChunkIds, [])
                      WHERE chunk_id IN $chunk_ids)
            WITH relationship,
                 [chunk_id IN coalesce(relationship.sourceChunkIds, [])
                  WHERE NOT chunk_id IN $chunk_ids] AS remaining_chunk_ids
            SET relationship.sourceChunkIds = remaining_chunk_ids,
                relationship.evidenceSentences = [],
                relationship.description = ''
            WITH relationship, remaining_chunk_ids
            WHERE size(remaining_chunk_ids) = 0
            DELETE relationship
            """,
            chunk_ids=chunk_ids,
        )

    session.run(
        """
        MATCH (chunk:Chunk)-[:PART_OF]->(document:Document)
        WHERE document.id IN $document_ids
        DETACH DELETE chunk
        """,
        document_ids=document_ids,
    )
    session.run(
        """
        MATCH (document:Document)
        WHERE document.id IN $document_ids
        DETACH DELETE document
        """,
        document_ids=document_ids,
    )

    if entity_ids:
        # Remove entities owned only by replaced documents. Entities still
        # mentioned by an unrelated document remain in the graph.
        session.run(
            """
            MATCH (entity:Entity)
            WHERE entity.id IN $entity_ids
              AND NOT EXISTS { MATCH (:Chunk)-[:MENTIONS]->(entity) }
            DETACH DELETE entity
            """,
            entity_ids=entity_ids,
        )


def replace_documents_atomic(
    session,
    document_ids: list[str],
    writer,
    *,
    clear_all: bool = False,
) -> None:
    """Replace documents and write new data in one Neo4j transaction.

    ``writer`` receives the transaction object and may use ``tx.run`` or pass
    it to the streaming graph builder.  Any exception rolls back both deletion
    and writes, leaving the previous graph intact.
    """
    def work(tx):
        if clear_all:
            tx.run("MATCH (n) DETACH DELETE n")
        else:
            replace_documents(tx, document_ids)
        writer(tx)

    if hasattr(session, "execute_write"):
        session.execute_write(work)
    elif hasattr(session, "write_transaction"):
        session.write_transaction(work)
    else:
        # Small fake sessions and older drivers can expose an explicit tx API.
        tx = session.begin_transaction()
        try:
            work(tx)
            tx.commit()
        except Exception:
            tx.rollback()
            raise


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
    """Upload a graph, replacing documents with matching stable IDs."""
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
        else:
            replace_documents(
                session,
                [node.get("id", "") for node in nodes if node.get("type") == "Document"],
            )

        # Create nodes
        logger.info(f"Uploading {len(nodes)} nodes...")
        for node in nodes:
            node_type = node.get("type", "Entity")
            node_id = node.get("id", "")
            provenance = {
                key: node[key]
                for key in (
                    "title", "url", "license", "source_domain", "scraped_at",
                    "crawler", "content_hash", "inferred_type",
                )
                if node.get(key) not in (None, "")
            }

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
                    SET n += $provenance
                    REMOVE n.entityType
                    """,
                    id=node_id,
                    source=source_str,
                    text=node.get("text", ""),
                    tokenCount=node.get("tokenCount", 0),
                    index=node.get("index", 0),
                    type=node_type,
                    provenance=provenance,
                )
            elif is_document:
                session.run(
                    """
                    MERGE (n:Document {id: $id})
                    SET n.name = $name,
                        n.type = $type,
                        n.description = $description,
                        n.source = $source,
                        n.chunk_count = $chunk_count
                    SET n += $provenance
                    REMOVE n.entityType
                    """,
                    id=node_id,
                    name=node.get("name", ""),
                    type=node_type,
                    description=node.get("description", ""),
                    source=node.get("source", []),
                    chunk_count=node.get("chunk_count", 0),
                    provenance=provenance,
                )
            else:
                session.run(
                    """
                    MERGE (n:Entity {id: $id})
                    SET n.name = $name,
                        n.type = $type,
                        n.description = $description,
                        n.importanceScore = $importance_score,
                        n.confidenceScore = $confidence_score,
                        n.embedding = $embedding
                    REMOVE n.entityType
                    """,
                    id=node_id,
                    name=node.get("name", ""),
                    type=node_type,
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
                    SET r.evidenceSentences = reduce(
                            items = coalesce(r.evidenceSentences, []),
                            item IN $evidence_sentences |
                            CASE WHEN item IN items THEN items ELSE items + [item] END
                        ),
                        r.sourceChunkIds = reduce(
                            ids = coalesce(r.sourceChunkIds, []),
                            chunk_id IN $source_chunk_ids |
                            CASE WHEN chunk_id IN ids THEN ids ELSE ids + [chunk_id] END
                        ),
                        r.description = CASE WHEN $description <> ''
                            THEN $description ELSE coalesce(r.description, '') END
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
            "RETURN n.id AS id, n.type AS type, "
            "n.description AS description, n.importanceScore AS importanceScore, "
            "n.confidenceScore AS confidenceScore, n.embedding AS embedding"
        )
        entity_nodes = [dict(record) for record in entity_result]

        # ── Fetch Document nodes ──
        doc_result = session.run(
            "MATCH (d:Document) "
            "RETURN d.id AS id, d.name AS name, d.type AS type, "
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
            "type": en.get("type", "Entity"),
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
            "type": dn.get("type", "Document"),
            "name": dn.get("name", dn["id"]),
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
            "type": en.get("type", "Entity"),
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
