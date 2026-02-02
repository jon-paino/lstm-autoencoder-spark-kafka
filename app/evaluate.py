"""
Evaluation and Visualization for LSTM Encoder-Decoder Anomaly Detection

Provides:
- Anomaly detection metrics (precision, recall, F1)
- Reconstruction visualizations
- Score distribution plots
- Training history plots
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from data_preprocessor import NYCTaxiPreprocessor, PreprocessorConfig, TimeSeriesDataset
from lstm_autoencoder import EncDecAD
from anomaly_scorer import AnomalyScorer
from train import load_model

logger = logging.getLogger(__name__)


def evaluate_detector(
    model: EncDecAD,
    scorer: AnomalyScorer,
    test_loader: DataLoader,
    test_week_info: List[Dict],
    device: torch.device
) -> Dict:
    """
    Evaluate anomaly detection performance.

    Metrics:
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        model: Trained LSTM encoder-decoder
        scorer: Fitted anomaly scorer with threshold
        test_loader: DataLoader with test sequences
        test_week_info: List of week metadata dicts
        device: Device for inference

    Returns:
        Dict with evaluation results and metrics
    """
    # Get predictions
    scores, errors = scorer.compute_scores(model, test_loader, device)
    predictions = scorer.predict(scores)

    # Ground truth
    actuals = np.array([w["is_anomaly"] for w in test_week_info])

    # Compute confusion matrix components
    true_positives = np.sum(predictions & actuals)
    false_positives = np.sum(predictions & ~actuals)
    false_negatives = np.sum(~predictions & actuals)
    true_negatives = np.sum(~predictions & ~actuals)

    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(predictions)

    results = {
        "scores": scores,
        "errors": errors,
        "predictions": predictions,
        "actuals": actuals,
        "week_info": test_week_info,
        "threshold": scorer.threshold,
        "metrics": {
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "true_negatives": int(true_negatives),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
        }
    }

    return results


def print_evaluation_report(results: Dict) -> None:
    """Print a formatted evaluation report."""
    metrics = results["metrics"]

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print("\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Normal  Anomaly")
    print(f"  Actual Normal    {metrics['true_negatives']:3d}      {metrics['false_positives']:3d}")
    print(f"  Actual Anomaly   {metrics['false_negatives']:3d}      {metrics['true_positives']:3d}")

    print("\nMetrics:")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1_score']:.2%}")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")

    print(f"\nThreshold: {results['threshold']:.2f}")

    print("\nPer-Week Results:")
    print(f"{'Week':<10} {'Score':>14} {'Predicted':>10} {'Actual':>10} {'Match':>6}")
    print("-" * 54)

    for score, pred, actual, week in zip(
        results["scores"],
        results["predictions"],
        results["actuals"],
        results["week_info"]
    ):
        pred_str = "ANOMALY" if pred else "normal"
        actual_str = "ANOMALY" if actual else "normal"
        match_str = "✓" if pred == actual else "✗"
        print(f"{week['year_week']:<10} {score:>14.2f} {pred_str:>10} {actual_str:>10} {match_str:>6}")


def plot_reconstruction(
    model: EncDecAD,
    sequence: np.ndarray,
    device: torch.device,
    title: str = "Reconstruction",
    save_path: Optional[str] = None
) -> Optional[object]:
    """
    Visualize original vs reconstructed sequence.

    Args:
        model: Trained model
        sequence: Input sequence, shape (seq_len,) or (seq_len, 1)
        device: Device for inference
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot")
        return None

    model.eval()

    # Prepare input
    if sequence.ndim == 1:
        sequence = sequence.reshape(-1, 1)

    x = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # (1, seq_len, 1)

    with torch.no_grad():
        x_reconstructed = model(x)

    original = sequence.squeeze()
    reconstructed = x_reconstructed.cpu().numpy().squeeze()
    error = np.abs(original - reconstructed)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Time axis (30-min intervals over a week)
    hours = np.arange(len(original)) * 0.5  # Hours

    # Original
    axes[0].plot(hours, original, label="Original", color="blue", linewidth=1)
    axes[0].set_ylabel("Normalized Value")
    axes[0].set_title(f"{title} - Original Sequence")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Reconstructed
    axes[1].plot(hours, reconstructed, label="Reconstructed", color="orange", linewidth=1)
    axes[1].set_ylabel("Normalized Value")
    axes[1].set_title("Reconstructed Sequence")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Error
    axes[2].fill_between(hours, 0, error, alpha=0.5, color="red", label="Error")
    axes[2].plot(hours, error, color="red", linewidth=0.5)
    axes[2].set_xlabel("Hours from Week Start")
    axes[2].set_ylabel("Absolute Error")
    axes[2].set_title(f"Reconstruction Error (Mean: {error.mean():.4f})")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Add day markers
    for ax in axes:
        for day in range(1, 7):
            ax.axvline(x=day * 24, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved reconstruction plot to {save_path}")

    return fig


def plot_score_distribution(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    test_actuals: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None
) -> Optional[object]:
    """
    Plot distribution of anomaly scores.

    Args:
        train_scores: Scores from training data
        test_scores: Scores from test data
        test_actuals: Boolean array of actual anomalies
        threshold: Anomaly threshold
        save_path: Optional path to save figure

    Returns:
        matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram of scores
    ax = axes[0]

    # Separate test scores by actual label
    normal_scores = test_scores[~test_actuals]
    anomaly_scores = test_scores[test_actuals]

    # Plot histograms
    bins = np.linspace(
        min(train_scores.min(), test_scores.min()),
        max(train_scores.max(), test_scores.max()),
        30
    )

    ax.hist(train_scores, bins=bins, alpha=0.5, label="Train (normal)", color="blue")
    ax.hist(normal_scores, bins=bins, alpha=0.5, label="Test (normal)", color="green")
    ax.hist(anomaly_scores, bins=bins, alpha=0.5, label="Test (anomaly)", color="red")

    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.0f}")

    ax.set_xlabel("Anomaly Score (Mahalanobis Distance)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Anomaly Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Scores over time
    ax = axes[1]

    x = np.arange(len(test_scores))
    colors = ["red" if a else "blue" for a in test_actuals]

    ax.bar(x, test_scores, color=colors, alpha=0.7)
    ax.axhline(y=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.0f}")

    ax.set_xlabel("Week Index")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Test Scores by Week (Blue=Normal, Red=Anomaly)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved score distribution plot to {save_path}")

    return fig


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
) -> Optional[object]:
    """
    Plot training and validation loss curves.

    Args:
        history: Training history dict with train_loss and val_loss
        save_path: Optional path to save figure

    Returns:
        matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validation Loss", color="orange", linewidth=2)

    if "best_epoch" in history:
        best_epoch = history["best_epoch"]
        best_val_loss = history["val_loss"][best_epoch - 1]
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch: {best_epoch}")
        ax.scatter([best_epoch], [best_val_loss], color="green", s=100, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale if loss varies a lot
    if history["train_loss"][0] / history["train_loss"][-1] > 10:
        ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training history plot to {save_path}")

    return fig


def plot_weekly_comparison(
    model: EncDecAD,
    sequences: np.ndarray,
    week_info: List[Dict],
    device: torch.device,
    save_path: Optional[str] = None
) -> Optional[object]:
    """
    Compare reconstructions for normal vs anomaly weeks.

    Args:
        model: Trained model
        sequences: Test sequences, shape (num_weeks, seq_len)
        week_info: List of week metadata
        device: Device for inference
        save_path: Optional path to save figure

    Returns:
        matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot")
        return None

    # Find one normal and one anomaly week
    normal_idx = next((i for i, w in enumerate(week_info) if not w["is_anomaly"]), None)
    anomaly_idx = next((i for i, w in enumerate(week_info) if w["is_anomaly"]), None)

    if normal_idx is None or anomaly_idx is None:
        logger.warning("Could not find both normal and anomaly weeks")
        return None

    model.eval()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for row, (idx, label) in enumerate([(normal_idx, "Normal"), (anomaly_idx, "Anomaly")]):
        seq = sequences[idx]
        week = week_info[idx]

        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)

        x = torch.FloatTensor(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            x_reconstructed = model(x)

        original = seq.squeeze()
        reconstructed = x_reconstructed.cpu().numpy().squeeze()
        error = np.abs(original - reconstructed)

        hours = np.arange(len(original)) * 0.5

        # Original
        axes[row, 0].plot(hours, original, color="blue", linewidth=1)
        axes[row, 0].set_title(f"{label} Week ({week['year_week']}) - Original")
        axes[row, 0].set_ylabel("Normalized Value")
        axes[row, 0].grid(True, alpha=0.3)

        # Reconstructed
        axes[row, 1].plot(hours, reconstructed, color="orange", linewidth=1)
        axes[row, 1].set_title("Reconstructed")
        axes[row, 1].grid(True, alpha=0.3)

        # Error
        axes[row, 2].fill_between(hours, 0, error, alpha=0.5, color="red")
        axes[row, 2].set_title(f"Error (Mean: {error.mean():.4f})")
        axes[row, 2].grid(True, alpha=0.3)

        if row == 1:
            for col in range(3):
                axes[row, col].set_xlabel("Hours")

    plt.suptitle("Normal vs Anomaly Week Comparison", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved weekly comparison plot to {save_path}")

    return fig


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM Encoder-Decoder Anomaly Detection"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained model artifacts"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/nyc_taxi.csv",
        help="Path to NYC taxi CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model and artifacts
    print("\nLoading model and artifacts...")
    model_dir = Path(args.model_dir)

    model = load_model(model_dir / "lstm_model.pt", device)
    scorer = AnomalyScorer.load(model_dir / "scorer.pkl")

    with open(model_dir / "training_history.pkl", "rb") as f:
        history = pickle.load(f)

    # Load preprocessor config (data split configuration used during training)
    config_path = model_dir / "preprocessor_config.pkl"
    if config_path.exists():
        with open(config_path, "rb") as f:
            preprocess_config = pickle.load(f)
        print(f"Loaded data split config: train={preprocess_config.train_weeks}, "
              f"val={preprocess_config.val_weeks}, threshold={preprocess_config.threshold_weeks}")
    else:
        logger.warning(f"No preprocessor_config.pkl found in {model_dir}, using defaults")
        preprocess_config = PreprocessorConfig()

    # Load and preprocess data using the SAME config as training
    print("\nLoading data...")
    preprocessor = NYCTaxiPreprocessor(config=preprocess_config)
    dataloaders, normalized_splits = preprocessor.preprocess(args.data_path, batch_size=1)

    # Compute train scores (for distribution plot)
    print("\nComputing scores...")
    train_scores, _ = scorer.compute_scores(model, dataloaders["train"], device)

    # Evaluate on test set
    test_week_info = preprocessor.get_test_week_info()
    results = evaluate_detector(
        model=model,
        scorer=scorer,
        test_loader=dataloaders["test"],
        test_week_info=test_week_info,
        device=device
    )

    # Print report
    print_evaluation_report(results)

    # Save results
    results_path = output_dir / "evaluation_results.pkl"
    with open(results_path, "wb") as f:
        # Convert numpy arrays for pickling
        save_results = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in results.items()
        }
        pickle.dump(save_results, f)
    print(f"\nSaved evaluation results to {results_path}")

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("\nGenerating plots...")

        # Training history
        plot_training_history(
            history,
            save_path=output_dir / "training_history.png"
        )

        # Score distribution
        plot_score_distribution(
            train_scores=train_scores,
            test_scores=results["scores"],
            test_actuals=results["actuals"],
            threshold=results["threshold"],
            save_path=output_dir / "score_distribution.png"
        )

        # Weekly comparison
        plot_weekly_comparison(
            model=model,
            sequences=normalized_splits["test"],
            week_info=test_week_info,
            device=device,
            save_path=output_dir / "weekly_comparison.png"
        )

        # Individual reconstructions for anomaly weeks
        for i, week in enumerate(test_week_info):
            if week["is_anomaly"]:
                plot_reconstruction(
                    model=model,
                    sequence=normalized_splits["test"][i],
                    device=device,
                    title=f"Week {week['year_week']}",
                    save_path=output_dir / f"reconstruction_{week['year_week']}.png"
                )

        print(f"\nPlots saved to {output_dir}/")
    elif not HAS_MATPLOTLIB:
        print("\nSkipping plots (matplotlib not available)")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
