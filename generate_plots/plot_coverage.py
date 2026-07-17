"""Plots for the KG fact coverage report (Method 1, Step 1.4).

Generates:
  1. Match rates grouped bar chart — exact / pair / mention rates
  2. Health score semicircular gauge
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive — works headless (Colab, CI)
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ── colour palette ────────────────────────────────────────────
GREEN  = "#2ecc71"
YELLOW = "#f1c40f"
ORANGE = "#e67e22"
RED    = "#e74c3c"
BLUE   = "#3498db"
PURPLE = "#9b59b6"
DARK   = "#2c3e50"
LIGHT  = "#ecf0f1"


def _score_color(rate: float) -> str:
    """Colour a rate: green ≥ 0.7, yellow ≥ 0.4, else red."""
    if rate >= 0.7:
        return GREEN
    elif rate >= 0.4:
        return ORANGE
    else:
        return RED


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def plot_coverage(
    coverage_path: str | Path,
    output_dir: str | Path = "output_eval/method1",
    show: bool = False,
) -> list[Path]:
    """Generate all coverage plots and return saved file paths.

    Args:
        coverage_path: Path to coverage_report.json.
        output_dir: Where to write the PNG files.
        show: If True, call plt.show() (only works interactively).

    Returns:
        List of Paths to saved plot files.
    """
    coverage_path = Path(coverage_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coverage_path) as f:
        data = json.load(f)

    saved: list[Path] = []

    # 1. Match rates bar chart
    path = output_dir / "coverage_bars.png"
    _plot_match_bars(data, path)
    saved.append(path)
    logger.info("  Coverage bars → %s", path)

    # 2. Health score gauge
    path = output_dir / "coverage_gauge.png"
    _plot_health_gauge(data, path)
    saved.append(path)
    logger.info("  Coverage gauge → %s", path)

    if show:
        plt.show()
    else:
        plt.close("all")

    return saved


# ═══════════════════════════════════════════════════════════════
# Individual plot functions
# ═══════════════════════════════════════════════════════════════

def _plot_match_bars(data: dict[str, Any], save_path: Path) -> None:
    """Grouped bar chart of exact, pair, and mention match rates."""
    rates = [
        data.get("exact_match_rate", 0),
        data.get("pair_match_rate", 0),
        data.get("mention_match_rate", 0),
    ]
    counts = [
        data.get("exact_match_count", 0),
        data.get("pair_match_count", 0),
        data.get("mention_match_count", 0),
    ]
    total = data.get("total_facts", 1)
    labels = ["Exact\nTriple", "Entity\nPair", "Entity\nMention"]
    colors = [_score_color(r) for r in rates]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, [r * 100 for r in rates], color=colors, width=0.55, edgecolor="white", linewidth=1.2)

    # Annotate bars with % and count
    for bar, rate, count in zip(bars, rates, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 1.5,
            f"{rate:.0%}\n({count}/{total})",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color=DARK,
        )

    ax.set_ylim(0, max(r * 100 for r in rates) * 1.35 + 5 if rates else 105)
    ax.set_ylabel("Match Rate (%)", fontsize=12, color=DARK)
    ax.set_title("KG Fact Coverage — Match Rates", fontsize=14, fontweight="bold", color=DARK, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors=DARK)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # Add a "partial match" annotation line if present
    partial_count = sum(
        1 for r in data.get("per_fact_results", [])
        if r.get("partial_match")
    )
    if partial_count > 0:
        ax.text(
            0.5, 0.97,
            f"🔍 {partial_count} facts have partial (substring) matches with KG nodes",
            transform=ax.transAxes, ha="center", fontsize=9,
            color=ORANGE, style="italic",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_health_gauge(data: dict[str, Any], save_path: Path) -> None:
    """Semicircular gauge showing the overall coverage health score."""
    score = data.get("health_score", 0)
    verdict = data.get("verdict", "")

    # Truncate verdict for display
    if len(verdict) > 80:
        verdict = verdict[:77] + "..."

    # Colour the gauge
    if score >= 80:
        color = GREEN
        band = "Excellent"
    elif score >= 60:
        color = YELLOW
        band = "Good"
    elif score >= 40:
        color = ORANGE
        band = "Fair"
    else:
        color = RED
        band = "Poor"

    fig, ax = plt.subplots(figsize=(6, 4.5), subplot_kw={"projection": None})

    # Draw the semicircle
    theta = np.linspace(np.pi, 0, 100)
    radius = 1.0

    # Background arc (gray)
    ax.fill_between(
        np.cos(theta) * radius, np.sin(theta) * radius,
        0, color="#e0e0e0", alpha=0.4,
    )

    # Filled arc proportional to score
    fill_theta = np.linspace(np.pi, np.pi * (1 - score / 100), 100)
    ax.fill_between(
        np.cos(fill_theta) * radius, np.sin(fill_theta) * radius,
        0, color=color, alpha=0.85,
    )

    # Score text
    ax.text(0, 0.15, f"{score:.0f}", fontsize=52, fontweight="bold",
            ha="center", va="center", color=DARK)
    ax.text(0, -0.12, f"/ 100", fontsize=16, ha="center", va="center", color="#7f8c8d")
    ax.text(0, -0.32, band, fontsize=14, fontweight="bold",
            ha="center", va="center", color=color)

    # Verdict below
    ax.text(0, -0.55, verdict, fontsize=8, ha="center", va="center",
            color="#7f8c8d", style="italic", wrap=True)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.75, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Fact Coverage Health Score", fontsize=14, fontweight="bold",
                 color=DARK, pad=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
