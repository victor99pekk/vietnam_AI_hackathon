"""Stage 5: Quality evaluation — re-exports from the standalone evaluation/ package.

The canonical evaluation modules now live in the top-level evaluation/ directory.
This module provides backward-compatible imports.
"""

import sys
from pathlib import Path

# Ensure the evaluation/ package is importable
_eval_dir = Path(__file__).resolve().parent.parent.parent.parent / "evaluation"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from evaluation.data_eval.metrics import QualityEvaluator   # noqa: E402, F811
from evaluation.data_eval import (                           # noqa: E402
    StructuralAuditor,
    load_kg_for_audit,
    SFTGenerator,
    TemplateSFTGenerator,
    SFTEvaluator,
)
from evaluation.model_eval import (                           # noqa: E402
    QADatasetGenerator,
    load_kg,
    load_raw_documents,
)

__all__ = [
    "QualityEvaluator",
    "StructuralAuditor",
    "load_kg_for_audit",
    "SFTGenerator",
    "TemplateSFTGenerator",
    "SFTEvaluator",
    "QADatasetGenerator",
    "load_kg",
    "load_raw_documents",
    "FineTuner",
    "FineTuneConfig",
    "AblationBenchmark",
]


def __getattr__(name):
    """Load optional model-training dependencies only when requested."""
    if name in {"FineTuner", "FineTuneConfig", "AblationBenchmark"}:
        import evaluation.model_eval as model_eval

        return getattr(model_eval, name)
    raise AttributeError(name)
