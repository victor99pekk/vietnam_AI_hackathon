"""Stage 5: Quality evaluation — re-exports from the standalone evaluation/ package.

The canonical evaluation modules live in the top-level evaluation/ directory.
When that directory is not on sys.path (e.g. in Colab after pip install),
falls back to the bundled copies in this package.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Try to import from the standalone evaluation/ package ────
_eval_dir = Path(__file__).resolve().parent.parent.parent.parent / "evaluation"
if str(_eval_dir) not in sys.path and _eval_dir.is_dir():
    sys.path.insert(0, str(_eval_dir))

try:
    from evaluation.data_eval.metrics import QualityEvaluator       # noqa: E402, F811
    from evaluation.data_eval import (                               # noqa: E402
        StructuralAuditor,
        load_kg_for_audit,
        SFTGenerator,
        SFTEvaluator,
    )
    from evaluation.model_eval import (                               # noqa: E402
        QADatasetGenerator,
        load_kg,
        load_raw_documents,
        FineTuner,
        FineTuneConfig,
    )
    from evaluation.model_eval.metrics import AblationBenchmark      # noqa: E402
except ImportError:
    logger.debug("evaluation/ package not on path — using bundled copies")
    from .metrics import QualityEvaluator                            # noqa: E402, F811
    from .structural_audit import (                                  # noqa: E402
        StructuralAuditor,
        load_kg_for_audit,
    )
    from .sft_generator import SFTGenerator                          # noqa: E402
    from .sft_evaluator import SFTEvaluator                          # noqa: E402
    from .dataset_gen import (                                       # noqa: E402
        QADatasetGenerator,
        load_kg,
        load_raw_documents,
    )
    from .finetune import FineTuner, FineTuneConfig                  # noqa: E402
    from .benchmark import AblationBenchmark                         # noqa: E402

__all__ = [
    "QualityEvaluator",
    "StructuralAuditor",
    "load_kg_for_audit",
    "SFTGenerator",
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
