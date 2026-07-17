"""Graph export to multiple output formats."""

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
        triples: list[tuple[str, str, str, str]],
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

    def _to_json(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
        output_dir: Path,
    ) -> Path:
        """Export as a single JSON file (node-link + entity/triple lists)."""
        path = output_dir / "knowledge_graph.json"
        graph_data = nx.node_link_data(graph)

        # Convert triples to serializable format
        triple_dicts = [
            {
                "subject": t[0],
                "predicate": t[1],
                "object": t[2],
                "source_text": t[3] if len(t) > 3 else "",
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
        """Export as GraphML (for tools like Gephi, Cytoscape)."""
        path = output_dir / "knowledge_graph.graphml"

        # Copy graph to avoid mutating with string conversions
        g = graph.copy()
        for _, data in g.nodes(data=True):
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
        with open(nodes_path, "w", encoding="utf-8") as f:
            f.write("id:ID,name,label,:LABEL\n")
            for node, data in graph.nodes(data=True):
                label = data.get("label", "Entity")
                f.write(f"{node},{node},{label},Entity;{label}\n")

        # Relationships CSV
        rels_path = neo4j_dir / "relationships.csv"
        with open(rels_path, "w", encoding="utf-8") as f:
            f.write(":START_ID,predicate,:END_ID,:TYPE\n")
            for subj, obj, data in graph.edges(data=True):
                predicates = data.get("predicates", ["related_to"])
                for pred in predicates:
                    f.write(f"{subj},{pred},{obj},RELATES_TO\n")

        logger.info(f"  Neo4j CSV -> {nodes_path}, {rels_path}")
        return [nodes_path, rels_path]

    def _to_rdf(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
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
