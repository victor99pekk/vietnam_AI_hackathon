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

    # Build a node-type lookup so relationship MATCH can use label-specific indexes
    id_to_type: dict[str, str] = {
        node["id"]: node.get("type", "Entity")
        for node in nodes
        if "id" in node
    }

    with driver.session() as session:
        if clear:
            logger.info("Clearing existing Neo4j data...")
            session.run("MATCH (n) DETACH DELETE n")
        else:
            replace_documents(
                session,
                [node.get("id", "") for node in nodes if node.get("type") == "Document"],
            )

        # ── Batched node uploads (UNWIND — one network round-trip per BATCH_SIZE) ──
        logger.info(f"Categorizing {len(nodes)} nodes for batched upload...")
        chunk_rows: list[dict] = []
        doc_rows: list[dict] = []
        entity_rows: list[dict] = []

        for node in nodes:
            node_type = node.get("type", "Entity")
            provenance = {
                key: node[key]
                for key in (
                    "title", "url", "license", "source_domain", "scraped_at",
                    "crawler", "content_hash", "inferred_type",
                )
                if node.get(key) not in (None, "")
            }

            if node_type == "Chunk":
                source_list = node.get("source", [])
                source_str = source_list[0] if isinstance(source_list, list) and source_list else source_list
                chunk_rows.append({
                    "id": node.get("id", ""),
                    "source": source_str,
                    "text": node.get("text", ""),
                    "tokenCount": node.get("tokenCount", 0),
                    "index": node.get("index", 0),
                    "type": node_type,
                    "provenance": provenance,
                })
            elif node_type == "Document":
                doc_rows.append({
                    "id": node.get("id", ""),
                    "name": node.get("name", ""),
                    "type": node_type,
                    "description": node.get("description", ""),
                    "source": node.get("source", []),
                    "chunk_count": node.get("chunk_count", 0),
                    "provenance": provenance,
                })
            else:
                entity_rows.append({
                    "id": node.get("id", ""),
                    "name": node.get("name", ""),
                    "type": node_type,
                    "description": node.get("description", ""),
                    "importance_score": node.get("importanceScore", 0.0),
                    "confidence_score": node.get("confidenceScore", 1.0),
                    "embedding": node.get("embedding"),
                })

        _upload_nodes_batched(session, "Chunk", chunk_rows)
        _upload_nodes_batched(session, "Document", doc_rows)
        _upload_nodes_batched(session, "Entity", entity_rows)

        # Create relationships (pre-aggregated in Python, batched with UNWIND)
        logger.info(f"Uploading {len(edges)} relationships...")
        _upload_relationships_batched(session, edges, id_to_type)

    driver.close()
    logger.info(f"Successfully uploaded {len(nodes)} nodes and {len(edges)} edges to Neo4j")


BATCH_SIZE = 5000
BATCH_SIZE_REL = 200000


def _upload_nodes_batched(session, label: str, rows: list[dict]) -> None:
    """Upload nodes of a single label in UNWIND batches."""
    if not rows:
        return
    logger.info(f"  Uploading {len(rows)} {label} nodes...")
    for offset in range(0, len(rows), BATCH_SIZE):
        batch = rows[offset : offset + BATCH_SIZE]
        if label == "Chunk":
            session.run(
                """
                UNWIND $batch AS item
                MERGE (n:Chunk {id: item.id})
                SET n.source = item.source, n.text = item.text,
                    n.tokenCount = item.tokenCount, n.index = item.index,
                    n.type = item.type
                SET n += item.provenance
                REMOVE n.entityType
                """, batch=batch,
            )
        elif label == "Document":
            session.run(
                """
                UNWIND $batch AS item
                MERGE (n:Document {id: item.id})
                SET n.name = item.name, n.type = item.type,
                    n.description = item.description, n.source = item.source,
                    n.chunk_count = item.chunk_count
                SET n += item.provenance
                REMOVE n.entityType
                """, batch=batch,
            )
        elif label == "Entity":
            session.run(
                """
                UNWIND $batch AS item
                MERGE (n:Entity {id: item.id})
                SET n.name = item.name, n.type = item.type,
                    n.description = item.description,
                    n.importanceScore = item.importance_score,
                    n.confidenceScore = item.confidence_score,
                    n.embedding = item.embedding
                REMOVE n.entityType
                """, batch=batch,
            )

# Label-filtered relationship patterns — one Cypher query per (src_label, tgt_label) combo
# so that Neo4j can use label-specific indexes on :Chunk(id), :Document(id), :Entity(id).
_LABEL_QUERIES: dict[tuple[str, str], str] = {
    ("Chunk", "Chunk"):     "MATCH (a:Chunk {id: row.source}), (b:Chunk {id: row.target})",
    ("Chunk", "Document"):  "MATCH (a:Chunk {id: row.source}), (b:Document {id: row.target})",
    ("Chunk", "Entity"):    "MATCH (a:Chunk {id: row.source}), (b:Entity {id: row.target})",
    ("Document", "Chunk"):  "MATCH (a:Document {id: row.source}), (b:Chunk {id: row.target})",
    ("Document", "Document"): "MATCH (a:Document {id: row.source}), (b:Document {id: row.target})",
    ("Document", "Entity"): "MATCH (a:Document {id: row.source}), (b:Entity {id: row.target})",
    ("Entity", "Chunk"):    "MATCH (a:Entity {id: row.source}), (b:Chunk {id: row.target})",
    ("Entity", "Document"): "MATCH (a:Entity {id: row.source}), (b:Document {id: row.target})",
    ("Entity", "Entity"):   "MATCH (a:Entity {id: row.source}), (b:Entity {id: row.target})",
}


def _upload_relationships_batched(
    session, edges: list[dict], id_to_type: dict[str, str]
) -> None:
    """Pre-aggregate duplicate relationships in Python, then upload in batches.

    Merges (source, target, predicate) keys in Python so Neo4j only sees each
    unique edge once per batch — no Cypher ``reduce`` loops, no duplicates.
    """
    from collections import defaultdict

    # Aggregate: (source, target, predicate, src_label, tgt_label) → metadata
    agg: dict[tuple[str, str, str, str, str], dict] = defaultdict(
        lambda: {"descriptions": set(), "evidence_sentences": set(), "source_chunk_ids": set()}
    )

    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        predicates = edge.get("predicates", ["related_to"])
        source_label = id_to_type.get(source, "Entity")
        target_label = id_to_type.get(target, "Entity")

        for pred in predicates:
            relation_records = [
                r for r in edge.get("relations", [])
                if r.get("predicate") == pred
            ]
            key = (source, target, pred, source_label, target_label)
            for r in relation_records:
                if r.get("description"):
                    agg[key]["descriptions"].add(r["description"])
                if r.get("evidence_sentence"):
                    agg[key]["evidence_sentences"].add(r["evidence_sentence"])
                if r.get("source_chunk_id"):
                    agg[key]["source_chunk_ids"].add(r["source_chunk_id"])

    # Bucket by (src_label, tgt_label) for static MATCH
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (source, target, pred, src_label, tgt_label), meta in agg.items():
        buckets[(src_label, tgt_label)].append({
            "source": source,
            "target": target,
            "predicate": pred,
            "evidence_sentences": list(meta["evidence_sentences"]),
            "source_chunk_ids": list(meta["source_chunk_ids"]),
            "description": list(meta["descriptions"])[0] if meta["descriptions"] else "",
        })

    total = sum(len(v) for v in buckets.values())
    uploaded = 0
    for (src_label, tgt_label), bucket_rows in buckets.items():
        match_clause = _LABEL_QUERIES.get(
            (src_label, tgt_label),
            f"MATCH (a:{src_label} {{id: row.source}}), (b:{tgt_label} {{id: row.target}})",
        )
        for offset in range(0, len(bucket_rows), BATCH_SIZE_REL):
            batch = bucket_rows[offset : offset + BATCH_SIZE_REL]
            uploaded += len(batch)
            logger.info(f"  [{src_label}→{tgt_label}] {uploaded - len(batch) + 1}–{uploaded} of {total}")

            # No Cypher reduce() loops — properties are already final lists
            query = f"""
            UNWIND $rows AS row
            {match_clause}
            CALL apoc.merge.relationship(a, row.predicate, {{}}, {{}}, b, {{}})
            YIELD rel
            SET rel.evidenceSentences = row.evidence_sentences,
                rel.sourceChunkIds = row.source_chunk_ids,
                rel.description = CASE WHEN row.description <> ''
                    THEN row.description ELSE coalesce(rel.description, '') END
            RETURN count(rel)
            """
            session.run(query, rows=batch)


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
    (kg_generator/evaluate/run_eval.py) from Neo4j. Reads connection details from
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
