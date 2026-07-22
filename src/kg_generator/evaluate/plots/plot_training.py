"""Plots for fine-tuning training metrics (Method 2 GPU/CPU).

Generates:
  1. Training loss curve per epoch/step
  2. Validation loss curve (if eval data used)
  3. Learning rate schedule
  4. Model comparison: training efficiency across models A/B/C
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

# ── Styling ───────────────────────────────────────────────────
COLORS = {
    "train": "#3498db",      # Blue
    "eval": "#e74c3c",       # Red
    "loss": "#95a5a6",       # Gray
}
STYLE = {
    "figure.facecolor": "#ecf0f1",
    "axes.facecolor": "#ffffff",
    "font.size": 10,
    "lines.linewidth": 2,
}
plt.rcParams.update(STYLE)


def plot_training_loss(
    trainer_state_path: str | Path,
    output_dir: str | Path = "output_eval/method2_gpu",
) -> Path:
    """Plot training/validation loss from trainer_state.json.
    
    Args:
        trainer_state_path: Path to trainer_state.json (from HuggingFace Trainer output)
        output_dir: Where to save plots
        
    Returns:
        Path to saved plot
    """
    trainer_state_path = Path(trainer_state_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not trainer_state_path.exists():
        logger.warning("trainer_state.json not found: %s", trainer_state_path)
        return None
    
    with open(trainer_state_path) as f:
        state = json.load(f)
    
    logs = state.get("log_history", [])
    if not logs:
        logger.warning("No training logs found in trainer_state.json")
        return None
    
    # Extract training and validation losses
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    learning_rates = []
    
    for entry in logs:
        if "loss" in entry:
            train_steps.append(entry.get("step", len(train_steps)))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_loss.append(entry["eval_loss"])
        if "learning_rate" in entry:
            learning_rates.append(entry["learning_rate"])
    
    # Plot training curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ── Loss curve ──
    ax = axes[0]
    if train_loss:
        ax.plot(train_steps, train_loss, marker="o", label="Training Loss", 
                color=COLORS["train"], alpha=0.8)
    if eval_loss:
        ax.plot(eval_steps, eval_loss, marker="s", label="Validation Loss", 
                color=COLORS["eval"], alpha=0.8)
    
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress: Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#ffffff")
    
    # ── Learning rate schedule ──
    ax = axes[1]
    if learning_rates:
        ax.plot(learning_rates, marker="o", color=COLORS["loss"], alpha=0.8)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    
    plt.tight_layout()
    plot_path = output_dir / f"training_loss.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info("Saved training loss plot → %s", plot_path)
    plt.close(fig)
    
    return plot_path


def plot_model_comparison_training(
    base_dir: str | Path = "output_eval/method2_gpu",
) -> list[Path]:
    """Compare training efficiency across models A/B/C.
    
    Looks for trainer_state.json in adapter directories under base_dir.
    
    Args:
        base_dir: Root directory containing model adapters
        
    Returns:
        List of saved plot paths
    """
    base_dir = Path(base_dir)
    output_dir = base_dir.parent
    
    if not base_dir.exists():
        logger.warning("Base directory not found: %s", base_dir)
        return []
    
    # Find all adapter directories with trainer_state.json
    adapters = {}
    for adapter_dir in sorted(base_dir.glob("*")):
        if not adapter_dir.is_dir():
            continue
        
        trainer_state = adapter_dir / "trainer_state.json"
        if not trainer_state.exists():
            continue
        
        # Extract model info (simplified from adapter name)
        model_name = adapter_dir.name
        adapters[model_name] = trainer_state
    
    if not adapters:
        logger.warning("No training logs found in %s", base_dir)
        return []
    
    plots = []
    
    # Plot individual training curves
    for model_name, trainer_state_path in adapters.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        
        with open(trainer_state_path) as f:
            state = json.load(f)
        
        logs = state.get("log_history", [])
        train_loss = []
        steps = []
        
        for entry in logs:
            if "loss" in entry:
                steps.append(entry.get("step", len(steps)))
                train_loss.append(entry["loss"])
        
        if train_loss:
            ax.plot(steps, train_loss, marker="o", color=COLORS["train"], alpha=0.8)
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training Progress: {model_name}")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#ffffff")
            
            plt.tight_layout()
            plot_path = output_dir / f"training_{model_name}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info("Saved training plot for %s → %s", model_name, plot_path)
            plots.append(plot_path)
        
        plt.close(fig)
    
    # Multi-model comparison (if >1 model)
    if len(adapters) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors_cycle = plt.cm.Set2(np.linspace(0, 1, len(adapters)))
        
        for idx, (model_name, trainer_state_path) in enumerate(adapters.items()):
            with open(trainer_state_path) as f:
                state = json.load(f)
            
            logs = state.get("log_history", [])
            train_loss = []
            steps = []
            
            for entry in logs:
                if "loss" in entry:
                    steps.append(entry.get("step", len(steps)))
                    train_loss.append(entry["loss"])
            
            if train_loss:
                ax.plot(steps, train_loss, marker="o", label=model_name, 
                       color=colors_cycle[idx], alpha=0.7)
        
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Model Training Comparison (GPU)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#ffffff")
        
        plt.tight_layout()
        plot_path = output_dir / "training_comparison.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info("Saved model comparison plot → %s", plot_path)
        plots.append(plot_path)
        plt.close(fig)
    
    return plots


def extract_training_metrics(
    trainer_state_path: str | Path,
) -> dict[str, Any]:
    """Extract summary metrics from trainer_state.json.
    
    Args:
        trainer_state_path: Path to trainer_state.json
        
    Returns:
        Dict with: final_loss, min_loss, epochs, total_steps, best_epoch
    """
    trainer_state_path = Path(trainer_state_path)
    
    if not trainer_state_path.exists():
        logger.warning("trainer_state.json not found: %s", trainer_state_path)
        return {}
    
    with open(trainer_state_path) as f:
        state = json.load(f)
    
    logs = state.get("log_history", [])
    train_losses = [entry["loss"] for entry in logs if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in logs if "eval_loss" in entry]
    
    metrics = {
        "final_train_loss": train_losses[-1] if train_losses else None,
        "min_train_loss": min(train_losses) if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "min_eval_loss": min(eval_losses) if eval_losses else None,
        "total_steps": state.get("global_step", 0),
        "training_seconds": state.get("max_steps", 0),  # Approximation
        "num_epochs": state.get("epoch", 0),
    }
    
    return metrics
