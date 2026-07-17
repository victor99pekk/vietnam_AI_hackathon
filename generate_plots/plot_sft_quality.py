"""Plots for the SFT quality report (Method 1, Step 1.3).

Generates:
  1. Quality metrics bar chart — faithfulness, relevancy, factual, diversity
  2. Bigram frequency horizontal bar chart
  3. Overall quality score gauge
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ── colour palette ────────────────────────────────────────────
FLAG_COLORS = {
    "green":  "#2ecc71",
    "yellow": "#f1c40f",
    "red":    "#e74c3c",
}
DEFAULT_BLUE = "#3498db"
DARK_BG     = "#2c3e50"
METRIC_COLORS = {
    "faithfulness":       "#e74c3c",
    "answer_relevancy":   "#3498db",
    "factual_correctness":"#2ecc71",
    "diversity":          "#9b59b6",
}


def _flag_color(flag: str) -> str:
    return FLAG_COLORS.get(flag, DEFAULT_BLUE)


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def plot_sft_quality(
    quality_path: str | Path,
    output_dir: str | Path = "output_eval/method1",
    show: bool = False,
) -> list[Path]:
    """Generate all SFT-quality plots and return saved file paths.

    Args:
        quality_path: Path to sft_quality_report.json.
        output_dir: Where to write the PNG files.
        show: If True, call plt.show() (only works interactively).

    Returns:
        List of Paths to saved plot files.
    """
    quality_path = Path(quality_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(quality_path) as f:
        data = json.load(f)

    saved: list[Path] = []

    # 1. Quality metrics bar chart
    path = output_dir / "sft_quality_bars.png"
    _plot_quality_bars(data, path)
    saved.append(path)
    logger.info("  Quality bars → %s", path)

    # 2. Bigram frequency chart
    path = output_dir / "sft_bigram_freq.png"
    _plot_bigram_freq(data, path)
    saved.append(path)
    logger.info("  Bigram freq → %s", path)

    # 3. Overall score gauge
    path = output_dir / "sft_quality_gauge.png"
    _plot_quality_gauge(data, path)
    saved.append(path)
    logger.info("  Quality gauge → %s", path)

    if show:
        plt.show()
    else:
        plt.close("all")

    return saved


# ═══════════════════════════════════════════════════════════════
# Individual plot functions
# ═══════════════════════════════════════════════════════════════

def _plot_quality_bars(data: dict[str, Any], save_path: Path) -> None:
    """Bar chart of SFT quality metrics."""
    metrics = data.get("metrics", {})

    # Extract numeric scores
    metric_items = [
        ("Faithfulness",        metrics.get("faithfulness", 0),        METRIC_COLORS["faithfulness"]),
        ("Answer Relevancy",    metrics.get("answer_relevancy", 0),    METRIC_COLORS["answer_relevancy"]),
        ("Factual Correctness", metrics.get("factual_correctness", 0), METRIC_COLORS["factual_correctness"]),
        ("Diversity",           metrics.get("semantic_diversity", {}).get("score", 0), METRIC_COLORS["diversity"]),
    ]

    labels = [m[0] for m in metric_items]
    values = [m[1] for m in metric_items]
    colors = [m[2] for m in metric_items]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, width=0.55)

    # Add threshold lines
    ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, linewidth=1, label="Good (0.7)")
    ax.axhline(y=0.4, color="orange", linestyle="--", alpha=0.5, linewidth=1, label="Fair (0.4)")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"SFT Quality Metrics  —  {data.get('pairs_evaluated', '?')} pairs evaluated",
        fontsize=14, fontweight="bold", color=DARK_BG,
    )

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold", color=DARK_BG)

    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add detail text below each bar
    detail_keys = ["faithfulness_detail", "relevancy_detail", "factual_detail", None]
    for i, (bar, dk) in enumerate(zip(bars, detail_keys)):
        if dk:
            detail = metrics.get(dk, "")
            if detail and len(detail) > 60:
                detail = detail[:57] + "…"
            ax.text(bar.get_x() + bar.get_width() / 2, -0.06,
                    detail if detail else "", ha="center", fontsize=7, color="grey",
                    rotation=0, va="top")

    note = metrics.get("note", "")
    if note:
        fig.text(0.5, 0.01, note, ha="center", fontsize=8, color="grey", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_bigram_freq(data: dict[str, Any], save_path: Path) -> None:
    """Horizontal bar chart of most common bigrams."""
    sem_div = data.get("metrics", {}).get("semantic_diversity", {})
    bigrams = sem_div.get("most_common_bigrams", [])

    if not bigrams:
        logger.warning("No bigram data found — skipping bigram plot")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No bigram data available", ha="center", va="center", fontsize=14, color="grey")
        ax.axis("off")
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    # Take top 15
    bigrams = bigrams[:15]
    labels = [b["bigram"].replace("_", " ") for b in bigrams]
    counts = [b["count"] for b in bigrams]

    # Reverse for horizontal bar (top at top)
    labels.reverse()
    counts.reverse()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Color gradient
    norm = plt.Normalize(min(counts), max(counts))
    cmap = plt.cm.viridis_r
    bar_colors = cmap(norm(counts))

    bars = ax.barh(labels, counts, color=bar_colors, edgecolor="white", linewidth=0.8)

    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_title(
        f"Most Common Bigrams  —  diversity score: {sem_div.get('score', 0):.2f}",
        fontsize=14, fontweight="bold", color=DARK_BG,
    )

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, fontweight="bold", color=DARK_BG)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_quality_gauge(data: dict[str, Any], save_path: Path) -> None:
    """Circular gauge for overall SFT quality score."""
    score = data.get("overall_score", 0)
    verdict = data.get("verdict", "")
    pairs = data.get("pairs_evaluated", 0)

    if score >= 0.8:
        color = FLAG_COLORS["green"]
    elif score >= 0.6:
        color = FLAG_COLORS["yellow"]
    elif score >= 0.4:
        color = "#e67e22"
    else:
        color = FLAG_COLORS["red"]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw a donut gauge
    wedges, texts = ax.pie(
        [score, max(0, 1.0 - score)],
        labels=[f"{score:.3f}", ""],
        colors=[color, "#ecf0f1"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.35, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 22, "fontweight": "bold", "color": DARK_BG},
    )

    # Center text
    ax.text(0, -0.08, f"{pairs} pairs", ha="center", fontsize=11, color="grey")
    ax.text(0, -0.20, "evaluated", ha="center", fontsize=11, color="grey")

    # Strip emoji from verdict for font compatibility
    import re
    clean_v = re.sub(r'[^\x00-\x7F]+', '', verdict).strip()
    short_v = clean_v[:70] + "…" if len(clean_v) > 70 else clean_v
    ax.set_title(short_v, fontsize=11, color=DARK_BG, pad=20)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
