"""
Step 2: Evaluate Model

Loads training artifacts from a directory and evaluates the model on the
held-out NYC taxi test set using window-level Mahalanobis scoring.
Defaults to `models/initial/` (the baseline output of step 1); pass
`--model-dir models/best` for the grid-sweep winner from step 4, or
`--model-dir models` for the prebuilt reference.

Reports precision / recall / F1, prints a per-week table, and saves
diagnostic plots to `evaluation/`. See `notebooks/model_design.ipynb`
for the underlying scoring methodology.

Usage:
    python code/2_evaluate_model.py
    python code/2_evaluate_model.py --model-dir models/best
    python code/2_evaluate_model.py --no-plots
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import EncDecAD, ModelConfig
from src.scorer import AnomalyScorer
from src.preprocess import (
    PreprocessorConfig,
    preprocess_pipeline,
    get_test_week_info,
)

# Register old module paths so pickled artifacts (saved with old names) can be loaded
import src.model, src.scorer, src.preprocess
sys.modules["lstm_autoencoder"] = src.model
sys.modules["anomaly_scorer"] = src.scorer
sys.modules["data_preprocessor"] = src.preprocess

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> EncDecAD:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EncDecAD(config=checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_window_level(model, scorer, test_loader, test_week_info, device):
    """Evaluate window-level Mahalanobis anomaly detection on the test set."""
    scores, errors = scorer.compute_scores(model, test_loader, device)
    predictions = scorer.predict(scores)
    actuals = np.array([w["is_anomaly"] for w in test_week_info])
    skip_mask = np.array([w.get("is_edge_case", False) for w in test_week_info])
    scored_mask = ~skip_mask

    tp = int(np.sum(predictions[scored_mask] & actuals[scored_mask]))
    fp = int(np.sum(predictions[scored_mask] & ~actuals[scored_mask]))
    fn = int(np.sum(~predictions[scored_mask] & actuals[scored_mask]))
    tn = int(np.sum(~predictions[scored_mask] & ~actuals[scored_mask]))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "scores": scores, "errors": errors, "predictions": predictions,
        "actuals": actuals, "skip_mask": skip_mask,
        "week_info": test_week_info, "threshold": scorer.threshold,
        "metrics": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": float(precision), "recall": float(recall), "f1": float(f1),
                    "n_scored": int(scored_mask.sum())},
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_training_history(history, save_path=None):
    """Plot training and validation loss curves."""
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validation Loss", color="orange", linewidth=2)
    if "best_epoch" in history:
        best = history["best_epoch"]
        ax.axvline(x=best, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch: {best}")
        ax.scatter([best], [history["val_loss"][best - 1]], color="green", s=100, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if history["train_loss"][0] / history["train_loss"][-1] > 10:
        ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_score_distribution(train_scores, test_scores, test_actuals, threshold, save_path=None):
    """Plot distribution of anomaly scores."""
    if not HAS_MATPLOTLIB:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    normal_scores = test_scores[~test_actuals]
    anomaly_scores = test_scores[test_actuals]
    bins = np.linspace(min(train_scores.min(), test_scores.min()), max(train_scores.max(), test_scores.max()), 30)

    axes[0].hist(train_scores, bins=bins, alpha=0.5, label="Train (normal)", color="blue")
    axes[0].hist(normal_scores, bins=bins, alpha=0.5, label="Test (normal)", color="green")
    axes[0].hist(anomaly_scores, bins=bins, alpha=0.5, label="Test (anomaly)", color="red")
    axes[0].axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.0f}")
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distributions")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = ["red" if a else "blue" for a in test_actuals]
    axes[1].bar(range(len(test_scores)), test_scores, color=colors, alpha=0.7)
    axes[1].axhline(y=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold")
    axes[1].set_xlabel("Week Index")
    axes[1].set_ylabel("Anomaly Score")
    axes[1].set_title("Test Scores by Week")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reconstruction(model, sequence, device, title="Reconstruction", save_path=None):
    """Visualize original vs reconstructed sequence."""
    if not HAS_MATPLOTLIB:
        return
    model.eval()
    if sequence.ndim == 1:
        sequence = sequence.reshape(-1, 1)
    x = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        x_recon = model(x)

    original = sequence.squeeze()
    reconstructed = x_recon.cpu().numpy().squeeze()
    error = np.abs(original - reconstructed)
    hours = np.arange(len(original)) * 0.5

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(hours, original, label="Original", color="blue", linewidth=1)
    axes[0].set_title(f"{title} - Original")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hours, reconstructed, label="Reconstructed", color="orange", linewidth=1)
    axes[1].set_title("Reconstructed")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(hours, 0, error, alpha=0.5, color="red")
    axes[2].set_title(f"Error (Mean: {error.mean():.4f})")
    axes[2].set_xlabel("Hours")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        for day in range(1, 7):
            ax.axvline(x=day * 24, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_weekly_comparison(model, sequences, week_info, device, save_path=None):
    """Compare reconstructions for normal vs anomaly weeks."""
    if not HAS_MATPLOTLIB:
        return
    normal_idx = next((i for i, w in enumerate(week_info) if not w["is_anomaly"]), None)
    anomaly_idx = next((i for i, w in enumerate(week_info) if w["is_anomaly"]), None)
    if normal_idx is None or anomaly_idx is None:
        return

    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for row, (idx, label) in enumerate([(normal_idx, "Normal"), (anomaly_idx, "Anomaly")]):
        seq = sequences[idx]
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)
        x = torch.FloatTensor(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            x_recon = model(x)

        original = seq.squeeze()
        reconstructed = x_recon.cpu().numpy().squeeze()
        error = np.abs(original - reconstructed)
        hours = np.arange(len(original)) * 0.5

        axes[row, 0].plot(hours, original, color="blue", linewidth=1)
        axes[row, 0].set_title(f"{label} ({week_info[idx]['year_week']}) - Original")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(hours, reconstructed, color="orange", linewidth=1)
        axes[row, 1].set_title("Reconstructed")
        axes[row, 1].grid(True, alpha=0.3)

        axes[row, 2].fill_between(hours, 0, error, alpha=0.5, color="red")
        axes[row, 2].set_title(f"Error (Mean: {error.mean():.4f})")
        axes[row, 2].grid(True, alpha=0.3)

        if row == 1:
            for c in range(3):
                axes[row, c].set_xlabel("Hours")

    plt.suptitle("Normal vs Anomaly Week Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM Anomaly Detection")
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models" / "initial"),
                        help="Directory containing lstm_model.pt, scaler.pkl, scorer.pkl, etc. "
                             "Defaults to models/initial/ (baseline from step 1). "
                             "Use --model-dir models/best for the grid-sweep winner from step 4, "
                             "or --model-dir models for the prebuilt reference.")
    parser.add_argument("--data-path", type=str, default=str(PROJECT_ROOT / "data" / "nyc_taxi.csv"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "evaluation"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load artifacts
    print("\nLoading model and artifacts...")
    model = load_model(model_dir / "lstm_model.pt", device)
    scorer = AnomalyScorer.load(str(model_dir / "scorer.pkl"))

    with open(model_dir / "training_history.pkl", "rb") as f:
        history = pickle.load(f)

    config_path = model_dir / "preprocessor_config.pkl"
    if config_path.exists():
        with open(config_path, "rb") as f:
            preprocess_config = pickle.load(f)
    else:
        preprocess_config = PreprocessorConfig()

    # Preprocess data with same config as training
    print("\nLoading data...")
    dataloaders, normalized_splits, scaler, week_info, split_indices = preprocess_pipeline(
        args.data_path, config=preprocess_config, batch_size=1
    )

    if scorer.mu is None or scorer.cov is None or scorer.threshold is None:
        raise RuntimeError(
            "Loaded scorer is missing window-level Mahalanobis state (mu/cov/threshold). "
            "Re-train with code/1_train_model.py (or run code/4_grid_sweep.py)."
        )

    # Compute train scores for distribution plot
    train_scores, _ = scorer.compute_scores(model, dataloaders["train"], device)
    test_week_info = get_test_week_info(week_info, split_indices)

    results = evaluate_window_level(model, scorer, dataloaders["test"], test_week_info, device)
    m = results["metrics"]

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"\nPrecision={m['precision']:.2%}  Recall={m['recall']:.2%}  F1={m['f1']:.2%}")
    print(f"(over {m['n_scored']} scored weeks)")

    print(f"\n{'Week':<12} {'Score':>14} {'Predicted':>10} {'Actual':>10} {'Match':>6}")
    print("-" * 56)
    for score, pred, actual, skip, week in zip(
        results["scores"], results["predictions"], results["actuals"],
        results["skip_mask"], test_week_info,
    ):
        if skip:
            continue
        pred_str = "ANOMALY" if pred else "normal"
        actual_str = "ANOMALY" if actual else "normal"
        match = "Y" if pred == actual else "N"
        print(f"{week['year_week']:<12} {score:>14.2f} {pred_str:>10} {actual_str:>10} {match:>6}")

    test_scores = results["scores"]
    test_actuals = results["actuals"]

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        plot_training_history(history, save_path=output_dir / "training_history.png")

        plot_score_distribution(train_scores, test_scores, test_actuals, scorer.threshold,
                                save_path=output_dir / "score_distribution.png")

        plot_weekly_comparison(model, normalized_splits["test"], test_week_info, device,
                               save_path=output_dir / "weekly_comparison.png")

        for i, week in enumerate(test_week_info):
            if week["is_anomaly"]:
                plot_reconstruction(model, normalized_splits["test"][i], device,
                                    title=f"Week {week['year_week']}",
                                    save_path=output_dir / f"reconstruction_{week['year_week']}.png")

        print(f"Plots saved to {output_dir}/")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
