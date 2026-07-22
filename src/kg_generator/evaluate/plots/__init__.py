"""Visualization utilities for KG evaluation metrics.

Generates plots from:
  - Method 1: structural_audit.json, sft_quality_report.json
  - Method 2: ablation benchmark results
"""

from kg_generator.evaluate.plots.plot_structural import plot_structural_audit
from kg_generator.evaluate.plots.plot_sft_quality import plot_sft_quality
from kg_generator.evaluate.plots.plot_ablation import plot_ablation

__all__ = ["plot_structural_audit", "plot_sft_quality", "plot_ablation"]
