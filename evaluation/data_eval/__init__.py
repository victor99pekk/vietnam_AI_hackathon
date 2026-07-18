"""Data Evaluation — is your KG generating high-quality SFT training data?

  metrics.py           → Basic KG quality (completeness, consistency, etc.)
  structural_audit.py  → Deep graph health (orphans, density, schema, dups)
  sft_generator.py     → Generate SFT instruction/response pairs from KG subgraphs
  sft_evaluator.py     → Score SFT quality via deepeval (faithfulness, relevancy, diversity)
  coverage.py          → Fact recall: how many source-document facts are in the KG?
"""

from evaluation.data_eval.metrics import QualityEvaluator
from evaluation.data_eval.structural_audit import StructuralAuditor, load_kg_for_audit
from evaluation.data_eval.neo4j_auditor import Neo4jStructuralAuditor
from evaluation.data_eval.sft_generator import SFTGenerator
from evaluation.data_eval.sft_evaluator import SFTEvaluator
from evaluation.data_eval.coverage import (
    FactExtractor,
    CoverageEvaluator,
    compute_kg_coverage,
)

__all__ = [
    "QualityEvaluator",
    "StructuralAuditor",
    "Neo4jStructuralAuditor",
    "load_kg_for_audit",
    "SFTGenerator",
    "SFTEvaluator",
    "FactExtractor",
    "CoverageEvaluator",
    "compute_kg_coverage",
]
