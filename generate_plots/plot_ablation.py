"""Plots for the fine-tuning ablation benchmark (Method 2).

Generates:
  1. Model comparison grouped bar chart — A (base) vs B (KG) vs C (raw)
  2. Per-metric radar chart across models
  3. Relative improvement chart — B and C gains over baseline A
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# ── colour palette ────────────────────────────────────────────
MODEL_COLORS = {
    "A (Base)":       "#95a5a6",
    "B (KG-Managed)": "#2ecc71",
    "C (Raw-Text)":   "#3498db",
}
MODEL_COLORS_SHORT = {
    "A": "#95a5a6",
    "B": "#2ecc71",
    "C": "#3498db",
}
DARK_BG = "#2c3e50"


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def plot_ablation(
    results_path: str | Path | None = None,
    output_dir: str | Path = "output_eval/method2",
    show: bool = False,
) -> list[Path]:
    """Generate all ablation benchmark plots.

    If results_path is None, tries to find benchmark JSON files in output_dir.

    Args:
        results_path: Path to a benchmark results JSON file (optional).
        output_dir: Where to write PNG files and find results if results_path is None.
        show: If True, call plt.show().

    Returns:
        List of Paths to saved plot files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to locate results file
    if results_path is None:
        candidates = sorted(
            output_dir.rglob("*benchmark*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ) + sorted(
            output_dir.rglob("*ablation*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ) + sorted(
            output_dir.rglob("*model*results*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not candidates:
            # Try the method2 combined results
            combined = output_dir / "method2_results.json"
            if combined.exists():
                candidates = [combined]
        if not candidates:
            logger.warning("No ablation results found in %s — generating placeholder plots", output_dir)
            saved = _plot_placeholder(output_dir)
            return saved
        results_path = candidates[0]

    results_path = Path(results_path)
    with open(results_path) as f:
        data = json.load(f)

    # Handle method2_results.json wrapper: {"ablation": {"models": {...}, "comparison": {...}}}
    if "ablation" in data and isinstance(data["ablation"], dict):
        data = data["ablation"]

    # Determine the structure — can be:
    #   {"models": {"A_base": {...}, "B_kg": {...}, "C_raw": {...}}, "comparison": {...}}
    #   {"models": {"A": {...}, "B": {...}, "C": {...}}, "comparison": {...}}
    #   or a flat dict with per-model keys
    models_data_raw = data.get("models", {})
    comparison = data.get("comparison", {})

    # Normalise model keys: "A_base" → "A", "B_kg" → "B", "C_raw" → "C", etc.
    models_data = _normalise_model_keys(models_data_raw)

    # If no nested "models" key, try to extract model results from flat dict
    if not models_data:
        models_data = _extract_models_from_flat(data)

    if not models_data:
        logger.warning("Could not parse model results from %s", results_path)
        saved = _plot_placeholder(output_dir)
        return saved

    saved: list[Path] = []

    # 1. Grouped bar chart
    path = output_dir / "ablation_bars.png"
    _plot_model_bars(models_data, comparison, path)
    saved.append(path)
    logger.info("  Ablation bars → %s", path)

    # 2. Radar chart comparing models
    path = output_dir / "ablation_radar.png"
    _plot_model_radar(models_data, path)
    saved.append(path)
    logger.info("  Ablation radar → %s", path)

    # 3. Relative improvement over baseline
    path = output_dir / "ablation_improvement.png"
    _plot_improvement(models_data, path)
    saved.append(path)
    logger.info("  Improvement → %s", path)

    if show:
        plt.show()
    else:
        plt.close("all")

    return saved


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _normalise_model_keys(
    models_data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Normalise model keys like 'A_base' → 'A', 'B_kg' → 'B', 'C_raw' → 'C'.

    Also filters out metadata fields (model name, total_predictions, avg_response_length)
    and keeps only numeric metric values.
    """
    # Fields to skip (not performance metrics)
    SKIP_FIELDS = {"model", "total_predictions", "avg_response_length"}

    normalised: dict[str, dict[str, float]] = {}
    for key, metrics in models_data.items():
        key_lower = key.lower()
        # Use the first character if it's a recognised model letter (a/b/c),
        # otherwise fall back to the full key
        first_char = key_lower[0] if key_lower else ""
        if first_char in ("a", "b", "c"):
            short_name = first_char.upper()
        else:
            short_name = key  # keep as-is if no match

        # Filter: only keep numeric metrics, skip metadata
        filtered: dict[str, float] = {}
        for mk, mv in metrics.items():
            if mk in SKIP_FIELDS:
                continue
            if mv is None:
                continue
            if isinstance(mv, (int, float)):
                filtered[mk] = float(mv)

        # Also store the display name from the "model" field if present
        if "model" in metrics:
            filtered["_display_name"] = metrics["model"]  # type: ignore[assignment]

        normalised[short_name] = filtered

    return normalised


def _extract_models_from_flat(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Try to extract per-model metrics from a flat-ish results dict."""
    models: dict[str, dict[str, float]] = {}

    # Known metric keys we care about
    metric_candidates = {
        "factual_accuracy", "multi_hop_accuracy", "hallucination_rate",
        "consistency", "perplexity", "f1_score", "exact_match",
        "precision", "recall", "accuracy",
    }

    # Look for keys like "model_a_factual_accuracy" or "A_factual_accuracy"
    model_labels = {
        "a": "A", "model_a": "A", "base": "A",
        "b": "B", "model_b": "B", "kg": "B", "kg_managed": "B",
        "c": "C", "model_c": "C", "raw": "C", "raw_text": "C",
    }

    for key, value in data.items():
        key_lower = key.lower()
        for prefix, model_name in model_labels.items():
            if key_lower.startswith(prefix + "_") or key_lower.startswith(prefix):
                metric_name = key_lower[len(prefix):].lstrip("_")
                if metric_name in metric_candidates or any(mc in metric_name for mc in metric_candidates):
                    if model_name not in models:
                        models[model_name] = {}
                    models[model_name][metric_name] = float(value) if isinstance(value, (int, float)) else 0.0

    return models


def _normalize_metric_name(name: str) -> str:
    """Convert snake_case metric names to display labels."""
    return name.replace("_", " ").title().replace(" ", "\n")


def _get_model_label(
    model_key: str,
    models_data: dict[str, dict[str, float]],
) -> str:
    """Get a human-readable label for a model.

    Uses the _display_name from the data if available, otherwise
    falls back to a hardcoded mapping.
    """
    model_metrics = models_data.get(model_key, {})
    if "_display_name" in model_metrics:
        # e.g. "A_base (Qwen2.5 base)" → keep as-is
        return str(model_metrics["_display_name"])

    # Fallback mapping
    fallback = {
        "A": "A (Base)",
        "B": "B (KG-Managed)",
        "C": "C (Raw-Text)",
    }
    return fallback.get(model_key, model_key)


# ═══════════════════════════════════════════════════════════════
# Individual plot functions
# ═══════════════════════════════════════════════════════════════

def _plot_model_bars(
    models_data: dict[str, dict[str, float]],
    comparison: dict[str, Any],
    save_path: Path,
) -> None:
    """Grouped bar chart: Model A vs B vs C across all metrics."""
    if not models_data:
        return

    # Collect all unique metric names across models
    all_metrics: list[str] = []
    for model_metrics in models_data.values():
        for m in model_metrics:
            if m not in all_metrics and not m.startswith("_"):
                all_metrics.append(m)

    if not all_metrics:
        logger.warning("No metrics found in model data")
        return

    model_names = sorted(models_data.keys())
    x = np.arange(len(all_metrics))
    width = 0.25
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(max(8, len(all_metrics) * 2), 6))

    for i, model_name in enumerate(model_names):
        values = [models_data[model_name].get(m, 0) for m in all_metrics]
        offset = (i - (n_models - 1) / 2) * width
        color = MODEL_COLORS_SHORT.get(model_name, MODEL_COLORS.get(
            f"{model_name} (KG-Managed)" if "B" in model_name else
            f"{model_name} (Raw-Text)" if "C" in model_name else
            f"{model_name} (Base)", "#95a5a6"
        ))
        label = _get_model_label(model_name, models_data)

        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="white", linewidth=0.8)

        # Value labels on top
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}" if val < 10 else f"{val:.1f}",
                        ha="center", fontsize=8, fontweight="bold", color=DARK_BG, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([_normalize_metric_name(m) for m in all_metrics], fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Ablation Benchmark — Model Comparison", fontsize=15, fontweight="bold", color=DARK_BG)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_model_radar(
    models_data: dict[str, dict[str, float]],
    save_path: Path,
) -> None:
    """Radar chart overlaying all three models."""
    if not models_data:
        return

    # Find common metrics across at least one model
    all_metrics: list[str] = []
    for mm in models_data.values():
        for m in mm:
            if m not in all_metrics and not m.startswith("_"):
                all_metrics.append(m)

    if len(all_metrics) < 3:
        logger.warning("Not enough metrics for radar chart (need ≥ 3)")
        return

    N = len(all_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_normalize_metric_name(m) for m in all_metrics], fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")

    for model_name in sorted(models_data.keys()):
        values = [models_data[model_name].get(m, 0) for m in all_metrics]
        values += values[:1]
        color = MODEL_COLORS_SHORT.get(model_name, "#95a5a6")
        label = _get_model_label(model_name, models_data)
        ax.fill(angles, values, alpha=0.08, color=color)
        ax.plot(angles, values, linewidth=2, color=color, marker="o", markersize=5, label=label)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Model Comparison — Radar", fontsize=14, fontweight="bold", color=DARK_BG, pad=25)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_improvement(
    models_data: dict[str, dict[str, float]],
    save_path: Path,
) -> None:
    """Bar chart showing B and C improvement over baseline A."""
    # Metrics where lower is better — we invert them so "improvement" always means "better"
    LOWER_IS_BETTER = {"hallucination_rate", "perplexity"}

    baseline = models_data.get("A", {})
    if not baseline:
        logger.warning("No baseline Model A data — skipping improvement plot")
        return

    metrics = [m for m in baseline if not m.startswith("_")]
    if not metrics:
        return

    b_data = models_data.get("B", {})
    c_data = models_data.get("C", {})

    b_label = _get_model_label("B", models_data)
    c_label = _get_model_label("C", models_data)

    b_improvements = []
    c_improvements = []
    metric_labels = []

    for m in metrics:
        base_val = baseline.get(m, 0)
        if base_val == 0:
            continue
        b_diff = b_data.get(m, 0) - base_val
        c_diff = c_data.get(m, 0) - base_val
        # Invert "lower is better" metrics so positive = improvement
        if m in LOWER_IS_BETTER:
            b_diff = -b_diff
            c_diff = -c_diff
        b_improvements.append((b_diff / base_val) * 100)
        c_improvements.append((c_diff / base_val) * 100)
        metric_labels.append(_normalize_metric_name(m))

    if not metric_labels:
        return

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(metric_labels) * 1.8), 5.5))

    bars_b = ax.bar(x - width / 2, b_improvements, width, label=b_label,
                     color=MODEL_COLORS_SHORT["B"], edgecolor="white", linewidth=0.8)
    bars_c = ax.bar(x + width / 2, c_improvements, width, label=c_label,
                     color=MODEL_COLORS_SHORT["C"], edgecolor="white", linewidth=0.8)

    # Zero line
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Value labels
    for bars in [bars_b, bars_c]:
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 0.3 if h >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f"{h:+.1f}%", ha="center", fontsize=8, fontweight="bold",
                    color="green" if h > 0 else "red", va=va)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("Improvement over Baseline (%)", fontsize=11)
    ax.set_title("Relative Improvement over Model A (Base)", fontsize=14, fontweight="bold", color=DARK_BG)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_placeholder(output_dir: Path) -> list[Path]:
    """Generate a placeholder image when no ablation data is available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ablation_placeholder.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.6, "No ablation results yet", ha="center", va="center",
            fontsize=18, fontweight="bold", color=DARK_BG)
    ax.text(0.5, 0.35, "Run Method 2 first:\npython evaluation/run_eval.py --method 2 model=a",
            ha="center", va="center", fontsize=12, color="grey")
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Placeholder → %s", path)
    return [path]
