"""
Step 4: Evaluate Model
Load trained artifacts, evaluate on test set, generate performance plots.

Usage:
    python code/4_evaluate_model.py
    python code/4_evaluate_model.py --no-plots
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
    ANOMALY_WINDOWS,
    preprocess_pipeline,
    get_test_week_info,
)

# Register old module paths so pickled artifacts (saved with old names) can be loaded
import src.model, src.scorer
sys.modules["lstm_autoencoder"] = src.model
sys.modules["anomaly_scorer"] = src.scorer

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


def evaluate_point_level(model, scorer, test_loader, test_week_info, device):
    """Evaluate point-level and window-level anomaly detection."""
    import pandas as pd

    point_scores, window_scores, _ = scorer.compute_point_scores(model, test_loader, device)
    point_predictions = scorer.predict_points(point_scores)
    window_predictions = scorer.predict_windows_from_points(point_predictions)
    window_actuals = np.array([w["is_anomaly"] for w in test_week_info])

    # Window-level metrics
    w_tp = np.sum(window_predictions & window_actuals)
    w_fp = np.sum(window_predictions & ~window_actuals)
    w_fn = np.sum(~window_predictions & window_actuals)
    w_tn = np.sum(~window_predictions & ~window_actuals)

    w_precision = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0
    w_recall = w_tp / (w_tp + w_fn) if (w_tp + w_fn) > 0 else 0
    w_f1 = 2 * w_precision * w_recall / (w_precision + w_recall) if (w_precision + w_recall) > 0 else 0

    # Point-level ground truth
    point_actuals = np.zeros_like(point_predictions, dtype=bool)
    for i, week in enumerate(test_week_info):
        for j in range(point_scores.shape[1]):
            point_time = pd.Timestamp(week["start_date"]) + pd.Timedelta(minutes=30 * j)
            for start, end in ANOMALY_WINDOWS:
                if pd.Timestamp(start) <= point_time <= pd.Timestamp(end):
                    point_actuals[i, j] = True
                    break

    p_tp = np.sum(point_predictions & point_actuals)
    p_fp = np.sum(point_predictions & ~point_actuals)
    p_fn = np.sum(~point_predictions & point_actuals)
    p_tn = np.sum(~point_predictions & ~point_actuals)

    p_precision = p_tp / (p_tp + p_fp) if (p_tp + p_fp) > 0 else 0
    p_recall = p_tp / (p_tp + p_fn) if (p_tp + p_fn) > 0 else 0
    p_f1 = 2 * p_precision * p_recall / (p_precision + p_recall) if (p_precision + p_recall) > 0 else 0

    return {
        "point_scores": point_scores,
        "point_predictions": point_predictions,
        "point_actuals": point_actuals,
        "window_scores": window_scores,
        "window_predictions": window_predictions,
        "window_actuals": window_actuals,
        "week_info": test_week_info,
        "point_threshold": scorer.point_threshold,
        "hard_criterion_k": scorer.config.hard_criterion_k,
        "point_metrics": {"tp": int(p_tp), "fp": int(p_fp), "fn": int(p_fn), "tn": int(p_tn),
                          "precision": float(p_precision), "recall": float(p_recall), "f1": float(p_f1)},
        "window_metrics": {"tp": int(w_tp), "fp": int(w_fp), "fn": int(w_fn), "tn": int(w_tn),
                           "precision": float(w_precision), "recall": float(w_recall), "f1": float(w_f1)},
    }


def evaluate_window_level(model, scorer, test_loader, test_week_info, device):
    """Evaluate window-level anomaly detection (legacy)."""
    scores, errors = scorer.compute_scores(model, test_loader, device)
    predictions = scorer.predict(scores)
    actuals = np.array([w["is_anomaly"] for w in test_week_info])

    tp = np.sum(predictions & actuals)
    fp = np.sum(predictions & ~actuals)
    fn = np.sum(~predictions & actuals)
    tn = np.sum(~predictions & ~actuals)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "scores": scores, "errors": errors, "predictions": predictions,
        "actuals": actuals, "week_info": test_week_info, "threshold": scorer.threshold,
        "metrics": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                    "precision": float(precision), "recall": float(recall), "f1": float(f1)},
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
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models"))
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

    use_point_level = scorer.mu_point is not None and scorer.point_threshold is not None
    scoring_mode = "point" if use_point_level else "window"
    print(f"\nScoring mode: {scoring_mode}-level")

    # Compute train scores for distribution plot
    train_scores, _ = scorer.compute_scores(model, dataloaders["train"], device)
    test_week_info = get_test_week_info(week_info, split_indices)

    # Evaluate
    if use_point_level:
        results = evaluate_point_level(model, scorer, dataloaders["test"], test_week_info, device)
        wm = results["window_metrics"]
        pm = results["point_metrics"]

        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"\nWindow-Level: Precision={wm['precision']:.2%}  Recall={wm['recall']:.2%}  F1={wm['f1']:.2%}")
        print(f"Point-Level:  Precision={pm['precision']:.2%}  Recall={pm['recall']:.2%}  F1={pm['f1']:.2%}")

        print(f"\n{'Week':<10} {'MaxScore':>12} {'AnomalyPts':>12} {'Predicted':>10} {'Actual':>10} {'Match':>6}")
        print("-" * 66)
        for i, week in enumerate(test_week_info):
            pred_str = "ANOMALY" if results["window_predictions"][i] else "normal"
            actual_str = "ANOMALY" if results["window_actuals"][i] else "normal"
            match = "Y" if results["window_predictions"][i] == results["window_actuals"][i] else "N"
            print(f"{week['year_week']:<10} {results['window_scores'][i]:>12.2f} "
                  f"{results['point_predictions'][i].sum():>12d} {pred_str:>10} {actual_str:>10} {match:>6}")

        test_scores = results["window_scores"]
        test_actuals = results["window_actuals"]
    else:
        results = evaluate_window_level(model, scorer, dataloaders["test"], test_week_info, device)
        m = results["metrics"]

        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"\nPrecision={m['precision']:.2%}  Recall={m['recall']:.2%}  F1={m['f1']:.2%}")

        print(f"\n{'Week':<10} {'Score':>14} {'Predicted':>10} {'Actual':>10} {'Match':>6}")
        print("-" * 54)
        for score, pred, actual, week in zip(results["scores"], results["predictions"], results["actuals"], test_week_info):
            pred_str = "ANOMALY" if pred else "normal"
            actual_str = "ANOMALY" if actual else "normal"
            match = "Y" if pred == actual else "N"
            print(f"{week['year_week']:<10} {score:>14.2f} {pred_str:>10} {actual_str:>10} {match:>6}")

        test_scores = results["scores"]
        test_actuals = results["actuals"]

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        plot_training_history(history, save_path=output_dir / f"training_history_{scoring_mode}.png")

        threshold = scorer.threshold if scorer.threshold else test_scores.max()
        plot_score_distribution(train_scores, test_scores, test_actuals, threshold,
                                save_path=output_dir / f"score_distribution_{scoring_mode}.png")

        plot_weekly_comparison(model, normalized_splits["test"], test_week_info, device,
                               save_path=output_dir / f"weekly_comparison_{scoring_mode}.png")

        for i, week in enumerate(test_week_info):
            if week["is_anomaly"]:
                plot_reconstruction(model, normalized_splits["test"][i], device,
                                    title=f"Week {week['year_week']}",
                                    save_path=output_dir / f"reconstruction_{week['year_week']}_{scoring_mode}.png")

        print(f"Plots saved to {output_dir}/")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
