#!/usr/bin/env python3
"""CLI entry point for generating evaluation plots.

Usage:
    # Generate all available plots
    python -m generate_plots

    # Plot specific method results
    python -m generate_plots --method 1
    python -m generate_plots --method 2

    # Specify custom paths
    python -m generate_plots --audit output_eval/method1/structural_audit.json
    python -m generate_plots --quality output_eval/method1/sft_quality_report.json
    python -m generate_plots --ablation output_eval/method2/benchmark_results.json
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_plots")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from KG evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m generate_plots                              # auto-detect and plot everything
  python -m generate_plots --method 1                   # structural + SFT quality plots
  python -m generate_plots --method 2                   # ablation plots only
  python -m generate_plots --audit path/to/structural_audit.json
  python -m generate_plots --output my_plots/
        """,
    )
    parser.add_argument(
        "--method", "-m",
        type=int,
        choices=[1, 2],
        default=None,
        help="Only generate plots for a specific method (1 or 2).",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Path to structural_audit.json (Method 1, Step 1.1).",
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=None,
        help="Path to sft_quality_report.json (Method 1, Step 1.3).",
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=None,
        help="Path to ablation/benchmark results JSON (Method 2).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Base output directory for plots (default: same dir as input).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (only works in a GUI environment).",
    )

    args = parser.parse_args()

    # ── Auto-detect paths if not specified ───────────────────
    output_eval = Path("output_eval")

    if args.audit is None and args.quality is None and args.ablation is None:
        # Auto-detect everything
        if args.method is None or args.method == 1:
            audit_default = output_eval / "method1" / "structural_audit.json"
            quality_default = output_eval / "method1" / "sft_quality_report.json"
            if audit_default.exists():
                args.audit = audit_default
            if quality_default.exists():
                args.quality = quality_default

        if args.method is None or args.method == 2:
            # ablation is auto-detected inside plot_ablation()
            args.ablation = args.ablation  # keep None — auto-detect

    # ── Generate plots ───────────────────────────────────────
    generated: list[Path] = []

    # Method 1: Structural audit
    if args.audit and args.audit.exists():
        from kg_generator.evaluate.plots.plot_structural import plot_structural_audit
        out_dir = args.output or args.audit.parent
        logger.info("=" * 50)
        logger.info("Generating structural audit plots…")
        generated += plot_structural_audit(args.audit, out_dir, show=args.show)

    # Method 1: SFT quality
    if args.quality and args.quality.exists():
        from kg_generator.evaluate.plots.plot_sft_quality import plot_sft_quality
        out_dir = args.output or args.quality.parent
        logger.info("=" * 50)
        logger.info("Generating SFT quality plots…")
        generated += plot_sft_quality(args.quality, out_dir, show=args.show)

    # Method 2: Ablation
    if args.method is None or args.method == 2:
        from kg_generator.evaluate.plots.plot_ablation import plot_ablation
        out_dir = args.output or (output_eval / "method2")
        logger.info("=" * 50)
        logger.info("Generating ablation plots…")
        generated += plot_ablation(args.ablation, out_dir, show=args.show)

    # ── Summary ──────────────────────────────────────────────
    if generated:
        logger.info("=" * 50)
        logger.info("✅ Generated %d plot(s):", len(generated))
        for p in generated:
            logger.info("   %s", p)
    else:
        logger.warning("No plots were generated.")
        logger.warning("Make sure you've run the evaluation first:")
        logger.warning("  make eval-method1 dataset=small")
        logger.warning("  make eval-method2 model=a   # (on Colab GPU)")
        sys.exit(1)


if __name__ == "__main__":
    main()
