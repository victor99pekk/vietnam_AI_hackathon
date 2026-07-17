"""KG Evaluation Pipeline — standalone package.

Two complementary evaluation types:

  data_eval  — Is my KG generating high-quality SFT training data?
    → structural audit + SFT generation + deepeval quality scoring

  model_eval — Does KG-structured data produce a better model?
    → dataset generation + LoRA fine-tuning + A/B/C model comparison

Usage:
    python evaluation/run_eval.py --method 1 --kg generated_KGs/output_debug/knowledge_graph.json
    python evaluation/run_eval.py --method 2 --kg generated_KGs/output_debug/knowledge_graph.json
    python evaluation/run_eval.py --method all --kg generated_KGs/output_debug/knowledge_graph.json
"""

from evaluation.data_eval.metrics import QualityEvaluator
from evaluation.data_eval import (
    StructuralAuditor,
    load_kg_for_audit,
    SFTGenerator,
    TemplateSFTGenerator,
    SFTEvaluator,
)
from evaluation.model_eval import (
    QADatasetGenerator,
    load_kg,
    load_raw_documents,
    FineTuner,
    FineTuneConfig,
)
from evaluation.model_eval.metrics import AblationBenchmark

__all__ = [
    # data_eval
    "QualityEvaluator",
    "StructuralAuditor",
    "load_kg_for_audit",
    "SFTGenerator",
    "TemplateSFTGenerator",
    "SFTEvaluator",
    # model_eval
    "QADatasetGenerator",
    "load_kg",
    "load_raw_documents",
    "FineTuner",
    "FineTuneConfig",
    "AblationBenchmark",
]
