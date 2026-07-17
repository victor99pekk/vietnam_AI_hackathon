"""Plots for the structural audit report (Method 1, Step 1.1).

Generates:
  1. Health score radar chart — 5 dimensions at a glance
  2. Category breakdown bar chart — green/yellow/red coloring
  3. Graph stats summary — nodes, edges, density side by side
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works headless (Colab, CI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def _flag_color(flag: str) -> str:
    return FLAG_COLORS.get(flag, DEFAULT_BLUE)


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def plot_structural_audit(
    audit_path: str | Path,
    output_dir: str | Path = "output_eval/method1",
    show: bool = False,
) -> list[Path]:
    """Generate all structural-audit plots and return saved file paths.

    Args:
        audit_path: Path to structural_audit.json.
        output_dir: Where to write the PNG files.
        show: If True, call plt.show() (only works interactively).

    Returns:
        List of Paths to saved plot files.
    """
    audit_path = Path(audit_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(audit_path) as f:
        data = json.load(f)

    saved: list[Path] = []

    # 1. Radar / spider chart
    path = output_dir / "structural_radar.png"
    _plot_radar(data, path)
    saved.append(path)
    logger.info("  Radar chart → %s", path)

    # 2. Category bar chart
    path = output_dir / "structural_bars.png"
    _plot_category_bars(data, path)
    saved.append(path)
    logger.info("  Category bars → %s", path)

    # 3. Graph stats summary
    path = output_dir / "structural_stats.png"
    _plot_graph_stats(data, path)
    saved.append(path)
    logger.info("  Graph stats → %s", path)

    # 4. Overall health gauge
    path = output_dir / "structural_gauge.png"
    _plot_health_gauge(data, path)
    saved.append(path)
    logger.info("  Health gauge → %s", path)

    if show:
        plt.show()
    else:
        plt.close("all")

    return saved


# ═══════════════════════════════════════════════════════════════
# Individual plot functions
# ═══════════════════════════════════════════════════════════════

def _plot_radar(data: dict[str, Any], save_path: Path) -> None:
    """Spider/radar chart of the 5 health dimensions."""
    categories = [
        "Orphans",
        "Density",
        "Schema",
        "Duplication",
        "Multi-hop",
    ]
    scores = [
        data["orphan_analysis"]["health_score"],
        data["density_analysis"]["health_score"],
        data["schema_compliance"]["health_score"],
        data["entity_duplication"]["health_score"],
        data["multi_hop_connectivity"]["health_score"],
    ]
    flags = [
        data["orphan_analysis"].get("flag", "green"),
        data["density_analysis"].get("flag", "green"),
        data["schema_compliance"].get("flag", "green"),
        data["entity_duplication"].get("flag", "green"),
        data["multi_hop_connectivity"].get("flag", "green"),
    ]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the loop
    scores_closed = scores + scores[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")

    # Draw y-axis grid
    ax.set_rlabel_position(30)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="grey")
    ax.set_ylim(0, 100)

    # Fill area
    ax.fill(angles, scores_closed, alpha=0.25, color=DEFAULT_BLUE)
    ax.plot(angles, scores_closed, linewidth=2, color=DEFAULT_BLUE, marker="o", markersize=6)

    # Color each point by flag
    for i, (angle, score, flag) in enumerate(zip(angles[:-1], scores, flags)):
        ax.plot(angle, score, "o", color=_flag_color(flag), markersize=10, markeredgecolor="white", markeredgewidth=1.5)

    overall = data.get("overall_health_score", 0)
    ax.set_title(
        f"KG Structural Health — {overall}/100",
        fontsize=14, fontweight="bold", pad=25, color=DARK_BG,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_category_bars(data: dict[str, Any], save_path: Path) -> None:
    """Horizontal bar chart with green/yellow/red per category."""
    categories_map = {
        "Orphan Rate":         ("orphan_analysis",         "orphan_rate",          False),
        "Density":             ("density_analysis",         "density",              False),
        "Schema Compliance":   ("schema_compliance",        "compliance_rate",      False),
        "Entity Duplication":  ("entity_duplication",       "duplicate_pair_count", True),
        "Multi-hop (2-hop %)": ("multi_hop_connectivity",   "reachable_2hop_pct",   False),
        "Multi-hop (3-hop %)": ("multi_hop_connectivity",   "reachable_3hop_pct",   False),
    }

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    health_scores: list[float] = []

    for label, (section, key, is_count) in categories_map.items():
        sec = data.get(section, {})
        val = sec.get(key, 0)
        health = sec.get("health_score", 100)
        flag = sec.get("flag", "green")

        labels.append(label)
        # Scale for display — percentages stay 0-1, counts get normalized
        if is_count and isinstance(val, (int, float)):
            # Show raw count as secondary label
            labels[-1] = f"{label}\n({int(val)} pairs)"
            values.append(health / 100.0)
        elif isinstance(val, float) and 0 <= val <= 1:
            values.append(val)
        else:
            values.append(health / 100.0)

        colors.append(_flag_color(flag))
        health_scores.append(health)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: raw metric values
    bars = ax1.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8, height=0.6)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Score / Rate", fontsize=11)
    ax1.set_title("Metric Values", fontsize=13, fontweight="bold", color=DARK_BG)
    ax1.invert_yaxis()
    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9, color="grey")

    # Right: health scores
    bars2 = ax2.barh(labels, health_scores, color=colors, edgecolor="white", linewidth=0.8, height=0.6)
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Health Score (0–100)", fontsize=11)
    ax2.set_title("Health Scores", fontsize=13, fontweight="bold", color=DARK_BG)
    ax2.invert_yaxis()
    ax2.axvline(x=80, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvline(x=60, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvline(x=40, color="red", linestyle="--", alpha=0.5, linewidth=1)
    for bar, hs in zip(bars2, health_scores):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{hs:.0f}", va="center", fontsize=10, fontweight="bold", color=DARK_BG)

    fig.suptitle("Structural Audit — Category Breakdown", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_graph_stats(data: dict[str, Any], save_path: Path) -> None:
    """Simple summary of graph-level statistics."""
    stats = data.get("graph_stats", {})
    num_nodes = stats.get("num_nodes", 0)
    num_edges = stats.get("num_edges", 0)
    num_triples = stats.get("num_triples", 0)
    density = stats.get("density", 0) if "density_analysis" not in data else data["density_analysis"].get("density", 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: nodes vs edges bar
    metrics = ["Nodes", "Edges", "Triples"]
    counts = [num_nodes, num_edges, num_triples]
    bar_colors = [DEFAULT_BLUE, "#e67e22", "#9b59b6"]
    bars = ax1.bar(metrics, counts, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Graph Size", fontsize=13, fontweight="bold", color=DARK_BG)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                 str(count), ha="center", fontsize=11, fontweight="bold", color=DARK_BG)

    # Right: density gauge-like display
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Draw a simple horizontal gauge
    density_val = float(density) if density else 0
    ax2.barh(0.5, density_val, height=0.15, color=_flag_color(
        "red" if density_val < 0.005 else ("yellow" if density_val > 0.2 else "green")
    ), edgecolor="white")
    ax2.barh(0.5, 1.0, height=0.15, color="lightgrey", zorder=0, alpha=0.3)
    ax2.text(0.5, 0.75, f"Density: {density_val:.5f}", ha="center", fontsize=18, fontweight="bold", color=DARK_BG)
    ax2.text(0.5, 0.3, "Ideal: 0.005–0.2", ha="center", fontsize=11, color="grey")

    fig.suptitle("KG Graph Statistics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_health_gauge(data: dict[str, Any], save_path: Path) -> None:
    """Semi-circular gauge showing overall health score."""
    score = data.get("overall_health_score", 0)
    verdict = data.get("verdict", "")

    # Determine color by score range
    if score >= 80:
        color = FLAG_COLORS["green"]
    elif score >= 60:
        color = FLAG_COLORS["yellow"]
    elif score >= 40:
        color = "#e67e22"  # orange
    else:
        color = FLAG_COLORS["red"]

    fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw={"projection": "polar"})

    # Draw only the top half
    theta = np.linspace(0, np.pi, 100)
    ax.fill_between(theta, 0, 100, alpha=0.08, color="lightgrey")

    # Color zones
    for start, end, zone_color, alpha in [
        (0, 40, FLAG_COLORS["red"], 0.15),
        (40, 60, "#e67e22", 0.15),
        (60, 80, FLAG_COLORS["yellow"], 0.15),
        (80, 100, FLAG_COLORS["green"], 0.15),
    ]:
        ax.fill_between(
            np.linspace(start / 100 * np.pi, end / 100 * np.pi, 30),
            0, 100, alpha=alpha, color=zone_color,
        )

    # Needle
    needle_angle = score / 100 * np.pi
    ax.plot([needle_angle, needle_angle], [0, 95], color=color, linewidth=3, zorder=10)
    ax.plot(needle_angle, 95, "o", color=color, markersize=12, zorder=11, markeredgecolor="white", markeredgewidth=2)

    ax.set_ylim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, np.pi * 0.25, np.pi * 0.5, np.pi * 0.75, np.pi])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=10, color="grey")
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.spines["polar"].set_visible(False)

    ax.set_title(f"Overall Health: {score}/100", fontsize=16, fontweight="bold", color=DARK_BG, pad=25)

    # Strip emoji from verdict for font compatibility
    import re
    clean_verdict = re.sub(r'[^\x00-\x7F]+', '', verdict).strip()
    short_verdict = clean_verdict[:80] + "…" if len(clean_verdict) > 80 else clean_verdict
    fig.text(0.5, 0.08, short_verdict, ha="center", fontsize=9, color="grey", style="italic")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
