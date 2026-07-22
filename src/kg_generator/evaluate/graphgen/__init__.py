"""GraphGen-style graph organization and QA synthesis.

This package is intentionally separate from the repository's older template and
evaluation generators so the methods can be compared without changing their
behaviour.
"""

from kg_generator.evaluate.graphgen.qa_generator import GraphGenQAGenerator
from kg_generator.evaluate.graphgen.subgraphs import (
    GraphGenSubgraphSampler,
    KnowledgeEdge,
    SamplingResult,
    load_graphgen_kg,
)

__all__ = [
    "GraphGenQAGenerator",
    "GraphGenSubgraphSampler",
    "KnowledgeEdge",
    "SamplingResult",
    "load_graphgen_kg",
]
