"""Graph export to multiple output formats."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class GraphExporter:
    """Exports knowledge graphs to various formats for downstream use."""

    def export(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, ...]],
        output_dir: Path,
        formats: list[str] | None = None,
    ) -> list[Path]:
        """Export to all requested formats, returning paths of written files."""
        formats = formats or ["json"]
        output_paths: list[Path] = []

        for fmt in formats:
            if fmt == "json":
                output_paths.append(self._to_json(graph, entities, triples, output_dir))
            elif fmt == "graphml":
                output_paths.append(self._to_graphml(graph, output_dir))
            elif fmt == "neo4j_csv":
                output_paths.extend(self._to_neo4j_csv(graph, entities, output_dir))
            elif fmt == "rdf":
                output_paths.append(self._to_rdf(graph, entities, triples, output_dir))
            elif fmt == "cytoscape":
                output_paths.append(self._to_cytoscape(graph, output_dir))
            else:
                logger.warning(f"Unknown export format: {fmt}")

        logger.info(f"Exported to {len(output_paths)} file(s) in {output_dir}")
        return output_paths

    @staticmethod
    def _normalize_node_props(data: dict[str, Any]) -> dict[str, Any]:
        """Filter node properties to match what Neo4j stores, per node type.

        This ensures .json and .graphml exports use the same attribute set
        as the Neo4j upload command.
        """
        node_type = data.get("type", "")
        cleaned: dict[str, Any] = {
            "id": data.get("id", ""),
            "type": node_type,
        }

        if node_type == "Chunk":
            source_list = data.get("source", [])
            if isinstance(source_list, list) and source_list:
                cleaned["source"] = source_list[0]
            else:
                cleaned["source"] = str(source_list) if source_list else ""
            cleaned["text"] = data.get("text", "")
            cleaned["tokenCount"] = data.get("tokenCount", 0)
            cleaned["index"] = data.get("index", 0)
        elif node_type == "Document":
            cleaned["name"] = data.get("name", data.get("id", ""))
            cleaned["description"] = data.get("description", "")
            cleaned["source"] = data.get("source", [])
            cleaned["chunk_count"] = data.get("chunk_count", 0)
        else:
            # Entity node
            cleaned["name"] = data.get("name", data.get("id", ""))
            cleaned["description"] = data.get("description", "")
            cleaned["importanceScore"] = data.get("importanceScore", 0.0)
            cleaned["confidenceScore"] = data.get("confidenceScore", 1.0)
            cleaned["embedding"] = data.get("embedding")

        return cleaned

    def _to_json(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, ...]],
        output_dir: Path,
    ) -> Path:
        """Export as a single JSON file (node-link + entity/triple lists).

        Node properties are normalized to match the Neo4j attribute schema.
        """
        path = output_dir / "knowledge_graph.json"
        graph_data = nx.node_link_data(graph)

        # Normalize node properties to match Neo4j schema
        for node in graph_data.get("nodes", []):
            normalized = self._normalize_node_props(node)
            node.clear()
            node.update(normalized)

        # Convert triples to serializable format
        triple_dicts = [
            {
                "subject": t[0],
                "predicate": t[1],
                "object": t[2],
                "evidence_sentence": t[3] if len(t) > 3 else "",
                "source_chunk_id": t[4] if len(t) > 4 else "",
                "description": t[5] if len(t) > 5 else "",
            }
            for t in triples
        ]

        output = {
            "graph": graph_data,
            "entities": entities,
            "triples": triple_dicts,
            "stats": {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "num_triples": len(triples),
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"  JSON -> {path}")
        return path

    def _to_graphml(self, graph: nx.DiGraph, output_dir: Path) -> Path:
        """Export as GraphML (for tools like Gephi, Cytoscape).

        Node properties are normalized to match the Neo4j attribute schema.
        """
        path = output_dir / "knowledge_graph.graphml"

        # Copy graph to avoid mutating the original
        g = graph.copy()

        # Normalize node properties to match Neo4j schema, then serialize
        for node, data in g.nodes(data=True):
            normalized = self._normalize_node_props({**data, "id": node})
            data.clear()
            data.update(normalized)
            for k, v in list(data.items()):
                if isinstance(v, (list, dict)):
                    data[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    data[k] = ""

        for _, _, data in g.edges(data=True):
            for k, v in list(data.items()):
                if isinstance(v, (list, dict)):
                    data[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    data[k] = ""

        nx.write_graphml(g, str(path))
        logger.info(f"  GraphML -> {path}")
        return path

    def _to_neo4j_csv(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        output_dir: Path,
    ) -> list[Path]:
        """Export as CSV files for Neo4j import (nodes.csv + relationships.csv)."""
        neo4j_dir = output_dir / "neo4j_import"
        neo4j_dir.mkdir(exist_ok=True)

        # Nodes CSV
        nodes_path = neo4j_dir / "nodes.csv"
        with open(nodes_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id:ID", "name", "type", ":LABEL"])
            for node, data in graph.nodes(data=True):
                node_type = data.get("type", "Entity")
                labels = node_type if node_type in ("Document", "Chunk") else "Entity"
                writer.writerow([node, data.get("name", ""), node_type, labels])

        # Relationships CSV
        rels_path = neo4j_dir / "relationships.csv"
        with open(rels_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ":START_ID",
                "predicate",
                ":END_ID",
                ":TYPE",
                "evidence_sentence",
                "source_chunk_id",
                "description",
            ])
            for subj, obj, data in graph.edges(data=True):
                relation_records = data.get("relations", [])
                if not relation_records:
                    relation_records = [
                        {"predicate": pred, "evidence_sentence": "", "source_chunk_id": ""}
                        for pred in data.get("predicates", ["related_to"])
                    ]
                for relation in relation_records:
                    pred = relation["predicate"]
                    writer.writerow([
                        subj,
                        pred,
                        obj,
                        pred.upper(),
                        relation.get("evidence_sentence", ""),
                        relation.get("source_chunk_id", ""),
                        relation.get("description", ""),
                    ])

        logger.info(f"  Neo4j CSV -> {nodes_path}, {rels_path}")
        return [nodes_path, rels_path]

    def _to_rdf(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, ...]],
        output_dir: Path,
    ) -> Path:
        """Export as RDF/Turtle format."""
        path = output_dir / "knowledge_graph.ttl"
        with open(path, "w", encoding="utf-8") as f:
            f.write("@prefix kg: <http://knowledge.graph/ontology/> .\n")
            f.write("@prefix ent: <http://knowledge.graph/entity/> .\n\n")

            for t in triples:
                s = t[0].replace(" ", "_").replace('"', "")
                o = t[2].replace(" ", "_").replace('"', "")
                p = t[1].replace(" ", "_").replace('"', "")
                f.write(f'ent:{s}\tkg:{p}\tent:{o} .\n')

        logger.info(f"  RDF/Turtle -> {path}")
        return path

    def _to_cytoscape(self, graph: nx.DiGraph, output_dir: Path) -> Path:
        """Export as Cytoscape.js JSON format."""
        path = output_dir / "knowledge_graph.cyjs.json"

        elements: list[dict] = []

        for node, data in graph.nodes(data=True):
            elements.append({
                "data": {
                    "id": node,
                    "label": data.get("label", "Entity"),
                    **{k: v for k, v in data.items() if k not in ("label",)},
                },
            })

        for subj, obj, data in graph.edges(data=True):
            elements.append({
                "data": {
                    "id": f"{subj}_{obj}",
                    "source": subj,
                    "target": obj,
                    "label": data.get("predicates", ["related_to"])[0],
                },
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"elements": elements}, f, indent=2, ensure_ascii=False)

        logger.info(f"  Cytoscape.js -> {path}")
        return path
