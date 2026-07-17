"""GraphGen-style graph organization and QA synthesis.

This package is intentionally separate from the repository's older template and
evaluation generators so the methods can be compared without changing their
behaviour.
"""

from evaluation.graphgen.qa_generator import GraphGenQAGenerator
from evaluation.graphgen.subgraphs import (
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
