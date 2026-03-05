"""
FCVAE Evaluation and Visualization

Generates per-combo visualizations for FCVAE model evaluation:
1. Reconstruction plots: original vs decoder mean (μ_x) with ±2σ_x confidence bands
2. Score distribution histograms with threshold line
3. Per-hour score heatmaps across test days
4. Training loss curves

Usage:
    python -m app.evaluate_fcvae --model-dir models/transactions_fcvae --output-dir results/fcvae
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, plotting functions disabled")


def plot_reconstruction(
    model: "FCVAE",
    windows: np.ndarray,
    scaler,
    device: torch.device,
    output_path: Path,
    combo_name: str,
    num_windows: int = 5,
    figsize: Tuple[int, int] = (15, 12),
) -> None:
    """
    Plot reconstruction with confidence bands.

    Shows original data vs decoder mean (μ_x) with ±2σ_x confidence bands.
    This is a key advantage of FCVAE: it outputs a distribution, not a point estimate.

    Args:
        model: Trained FCVAE model
        windows: Array of shape (N, W) - normalized windows
        scaler: Fitted scaler for inverse transform
        device: Torch device
        output_path: Directory to save plot
        combo_name: Name for plot title
        num_windows: Number of windows to plot
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping reconstruction plot")
        return

    model.eval()
    num_windows = min(num_windows, len(windows))

    fig, axes = plt.subplots(num_windows, 1, figsize=figsize)
    if num_windows == 1:
        axes = [axes]

    hours = np.arange(24)

    for idx, ax in enumerate(axes):
        window = windows[idx]

        # Get reconstruction
        x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            mu_x, var_x = model.reconstruct(x)
            mu_x = mu_x.squeeze().cpu().numpy()
            var_x = var_x.squeeze().cpu().numpy()

        std_x = np.sqrt(var_x)

        # Inverse transform for original scale (optional, use normalized for cleaner plots)
        # For clarity, plot in normalized space

        # Plot original
        ax.plot(hours, window, 'b-', linewidth=2, label='Original', marker='o', markersize=4)

        # Plot reconstruction mean
        ax.plot(hours, mu_x, 'r--', linewidth=2, label='Reconstruction (μ_x)')

        # Plot confidence bands
        ax.fill_between(
            hours,
            mu_x - 2 * std_x,
            mu_x + 2 * std_x,
            alpha=0.3,
            color='red',
            label='±2σ_x confidence'
        )

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Normalized Count')
        ax.set_title(f'{combo_name} - Window {idx + 1}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))

    plt.tight_layout()
    save_path = output_path / f"{combo_name.replace('/', '_')}_reconstruction.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved reconstruction plot to {save_path}")


def plot_score_distributions(
    point_scores: np.ndarray,
    window_scores: np.ndarray,
    point_threshold: float,
    window_threshold: Optional[float],
    output_path: Path,
    combo_name: str,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Plot score distribution histograms with threshold lines.

    Note: For FCVAE, anomalies have LOWER scores, so they appear on the LEFT.

    Args:
        point_scores: Per-point NLL scores (N, T)
        window_scores: Per-window mean scores (N,)
        point_threshold: Point-level threshold
        window_threshold: Window-level threshold (optional)
        output_path: Directory to save plot
        combo_name: Name for plot title
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping score distribution plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Point-level score distribution
    ax = axes[0]
    flat_scores = point_scores.flatten()
    ax.hist(flat_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(
        point_threshold, color='red', linestyle='--', linewidth=2,
        label=f'Threshold: {point_threshold:.2f}'
    )
    ax.set_xlabel('Point NLL Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{combo_name} - Point Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation for anomaly region
    ax.annotate(
        '← Anomalies',
        xy=(point_threshold - 0.1, ax.get_ylim()[1] * 0.9),
        fontsize=10, color='red'
    )

    # Window-level score distribution
    ax = axes[1]
    ax.hist(window_scores, bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
    if window_threshold is not None:
        ax.axvline(
            window_threshold, color='red', linestyle='--', linewidth=2,
            label=f'Threshold: {window_threshold:.2f}'
        )
    ax.set_xlabel('Window Mean NLL Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{combo_name} - Window Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_path / f"{combo_name.replace('/', '_')}_score_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved score distribution plot to {save_path}")


def plot_score_heatmap(
    point_scores: np.ndarray,
    point_threshold: float,
    output_path: Path,
    combo_name: str,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot per-hour score heatmap across windows/days.

    X-axis: Hour of day (0-23)
    Y-axis: Window index (or day index)
    Color: Point-level NLL score

    Useful for spotting systematic hour-of-day effects.

    Args:
        point_scores: Per-point NLL scores (N, 24)
        point_threshold: Point-level threshold
        output_path: Directory to save plot
        combo_name: Name for plot title
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping score heatmap")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        point_scores,
        aspect='auto',
        cmap='RdYlGn',  # Red=low (anomalous), Green=high (normal)
        interpolation='nearest'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NLL Score (lower = anomalous)')

    # Mark threshold on colorbar
    cbar.ax.axhline(y=point_threshold, color='black', linestyle='--', linewidth=2)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Window Index')
    ax.set_title(f'{combo_name} - Score Heatmap')
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f'{h}:00' for h in range(0, 24, 3)])

    plt.tight_layout()
    save_path = output_path / f"{combo_name.replace('/', '_')}_score_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved score heatmap to {save_path}")


def plot_training_history(
    history: Dict,
    output_path: Path,
    combo_name: str,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plot training loss curves.

    Shows train vs validation ELBO loss per epoch with learning rate overlay.

    Args:
        history: Training history dict with 'train_loss', 'val_loss', 'learning_rates'
        output_path: Directory to save plot
        combo_name: Name for plot title
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping training history plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)

    # Mark best epoch
    best_epoch = history.get('best_epoch', np.argmin(history['val_loss']) + 1)
    best_val_loss = history['val_loss'][best_epoch - 1]
    ax.scatter([best_epoch], [best_val_loss], marker='*', s=200, c='gold',
               edgecolors='black', zorder=5, label=f'Best (epoch {best_epoch})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('ELBO Loss')
    ax.set_title(f'{combo_name} - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate schedule
    ax = axes[1]
    if 'learning_rates' in history and history['learning_rates']:
        ax.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'{combo_name} - Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No LR data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{combo_name} - Learning Rate Schedule')

    plt.tight_layout()
    save_path = output_path / f"{combo_name.replace('/', '_')}_training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training history plot to {save_path}")


def plot_combo_comparison(
    registry: "FCVAERegistry",
    all_scores: Dict[Tuple[str, str], Dict],
    output_path: Path,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Plot 2x2 comparison grid of all combos.

    Each cell shows score distribution with threshold.

    Args:
        registry: FCVAERegistry with models and scorers
        all_scores: Dict mapping combo to {'point_scores', 'window_scores'}
        output_path: Directory to save plot
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping combo comparison plot")
        return

    from app.transaction_config import COMBO_KEYS

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, combo in enumerate(COMBO_KEYS):
        ax = axes[idx]

        if combo not in all_scores:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{combo[0]}/{combo[1]}')
            continue

        scores = all_scores[combo]
        window_scores = scores['window_scores']
        scorer = registry.get_scorer(combo)

        ax.hist(window_scores, bins=25, alpha=0.7, color='steelblue', edgecolor='black')

        if scorer.window_threshold is not None:
            ax.axvline(
                scorer.window_threshold, color='red', linestyle='--', linewidth=2,
                label=f'τ={scorer.window_threshold:.2f}'
            )

        # Count anomalies
        n_anomalies = np.sum(window_scores < scorer.window_threshold) if scorer.window_threshold else 0
        ax.set_title(f'{combo[0]}/{combo[1]} ({n_anomalies} anomalies)')
        ax.set_xlabel('Window Score')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('FCVAE Window Score Distributions by Combo', fontsize=14)
    plt.tight_layout()
    save_path = output_path / "combo_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved combo comparison plot to {save_path}")


def plot_anomaly_reconstruction(
    model: "FCVAE",
    normal_window: np.ndarray,
    anomaly_window: np.ndarray,
    anomaly_labels: np.ndarray,
    point_threshold: float,
    device: torch.device,
    output_path: Path,
    combo_name: str,
    anomaly_type: str = "spike",
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    """
    Plot reconstruction comparison: normal window vs anomaly-injected window.

    Shows:
    - Row 1: Original windows (normal vs anomaly) with reconstruction and confidence bands
    - Row 2: Per-point NLL scores with threshold line, highlighting injected hours

    This visualization helps understand how the model scores injected anomalies.

    Args:
        model: Trained FCVAE model
        normal_window: Normal window array (W,) - normalized
        anomaly_window: Anomaly-injected window array (W,) - normalized
        anomaly_labels: Per-point binary labels (W,) - 1 for injected hours
        point_threshold: Point-level NLL threshold
        device: Torch device
        output_path: Directory to save plot
        combo_name: Name for plot title
        anomaly_type: Type of anomaly ("spike" or "dip")
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping anomaly reconstruction plot")
        return

    model.eval()
    hours = np.arange(len(normal_window))

    # Get reconstructions for both windows
    with torch.no_grad():
        # Normal window
        x_normal = torch.FloatTensor(normal_window).unsqueeze(0).unsqueeze(0).to(device)
        mu_normal, var_normal = model.reconstruct(x_normal)
        mu_normal = mu_normal.squeeze().cpu().numpy()
        var_normal = var_normal.squeeze().cpu().numpy()
        std_normal = np.sqrt(var_normal)

        # Compute NLL scores for normal
        nll_normal = -0.5 * (np.log(var_normal) + (normal_window - mu_normal) ** 2 / var_normal)

        # Anomaly window
        x_anomaly = torch.FloatTensor(anomaly_window).unsqueeze(0).unsqueeze(0).to(device)
        mu_anomaly, var_anomaly = model.reconstruct(x_anomaly)
        mu_anomaly = mu_anomaly.squeeze().cpu().numpy()
        var_anomaly = var_anomaly.squeeze().cpu().numpy()
        std_anomaly = np.sqrt(var_anomaly)

        # Compute NLL scores for anomaly
        nll_anomaly = -0.5 * (np.log(var_anomaly) + (anomaly_window - mu_anomaly) ** 2 / var_anomaly)

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Row 1, Col 1: Normal window reconstruction
    ax = axes[0, 0]
    ax.plot(hours, normal_window, 'b-', linewidth=2, label='Original', marker='o', markersize=4)
    ax.plot(hours, mu_normal, 'r--', linewidth=2, label='Reconstruction (μ_x)')
    ax.fill_between(hours, mu_normal - 2 * std_normal, mu_normal + 2 * std_normal,
                    alpha=0.3, color='red', label='±2σ_x')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Normalized Count')
    ax.set_title(f'{combo_name} - Normal Window')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(hours) - 1)

    # Row 1, Col 2: Anomaly window reconstruction
    ax = axes[0, 1]
    ax.plot(hours, anomaly_window, 'b-', linewidth=2, label='Original (with injection)', marker='o', markersize=4)
    ax.plot(hours, mu_anomaly, 'r--', linewidth=2, label='Reconstruction (μ_x)')
    ax.fill_between(hours, mu_anomaly - 2 * std_anomaly, mu_anomaly + 2 * std_anomaly,
                    alpha=0.3, color='red', label='±2σ_x')

    # Highlight injected hours
    injected_mask = anomaly_labels > 0
    if np.any(injected_mask):
        ax.scatter(hours[injected_mask], anomaly_window[injected_mask],
                   c='orange', s=100, zorder=5, marker='s', label='Injected hours')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Normalized Count')
    ax.set_title(f'{combo_name} - {anomaly_type.title()} Injection')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(hours) - 1)

    # Row 2, Col 1: Normal window NLL scores
    ax = axes[1, 0]
    bars = ax.bar(hours, nll_normal, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(point_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {point_threshold:.2f}')
    ax.set_xlabel('Hour')
    ax.set_ylabel('NLL Score')
    ax.set_title(f'Normal Window - Point NLL Scores (mean: {np.mean(nll_normal):.3f})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Color bars below threshold
    for i, bar in enumerate(bars):
        if nll_normal[i] < point_threshold:
            bar.set_color('red')

    # Row 2, Col 2: Anomaly window NLL scores
    ax = axes[1, 1]
    colors = ['orange' if anomaly_labels[i] > 0 else 'steelblue' for i in range(len(hours))]
    bars = ax.bar(hours, nll_anomaly, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(point_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {point_threshold:.2f}')
    ax.set_xlabel('Hour')
    ax.set_ylabel('NLL Score')
    ax.set_title(f'{anomaly_type.title()} Window - Point NLL Scores (mean: {np.mean(nll_anomaly):.3f})')

    # Add legend explaining colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', label='Injected hours'),
        Patch(facecolor='steelblue', label='Normal hours'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Threshold: {point_threshold:.2f}')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Color bars below threshold red (except injected which stay orange)
    for i, bar in enumerate(bars):
        if nll_anomaly[i] < point_threshold and anomaly_labels[i] == 0:
            bar.set_color('red')

    # Count anomalous points
    n_anomalous_normal = np.sum(nll_normal < point_threshold)
    n_anomalous_injected = np.sum(nll_anomaly < point_threshold)
    injected_below = np.sum((nll_anomaly < point_threshold) & (anomaly_labels > 0))
    total_injected = np.sum(anomaly_labels > 0)

    plt.suptitle(
        f'{combo_name} - Anomaly Reconstruction Comparison\n'
        f'Normal: {n_anomalous_normal}/24 points below threshold | '
        f'Anomaly: {n_anomalous_injected}/24 below threshold ({injected_below}/{total_injected} injected detected)',
        fontsize=12
    )

    plt.tight_layout()
    save_path = output_path / f"{combo_name.replace('/', '_')}_anomaly_{anomaly_type}_reconstruction.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved anomaly reconstruction plot to {save_path}")


def get_daily_ground_truth(
    preprocessor: "TransactionPreprocessor",
    combo: Tuple[str, str],
    test_start_day: int,
    num_test_days: int,
) -> np.ndarray:
    """
    Extract ground truth daily anomaly labels from the data.

    A day is considered anomalous if ANY transaction in that day has is_anomaly=1.

    Args:
        preprocessor: TransactionPreprocessor with loaded data
        combo: (network_type, txn_type) tuple
        test_start_day: First day index of test period
        num_test_days: Number of test days

    Returns:
        Binary array (num_test_days,) where True = anomalous day
    """
    hourly_df = preprocessor.combo_hourly.get(combo)
    if hourly_df is None:
        return np.zeros(num_test_days, dtype=bool)

    # Check if is_anomaly column exists
    if "is_anomaly" not in hourly_df.columns:
        return np.zeros(num_test_days, dtype=bool)

    daily_ground_truth = np.zeros(num_test_days, dtype=bool)

    for day_offset in range(num_test_days):
        day_idx = test_start_day + day_offset
        day_start_hour = day_idx * 24
        day_end_hour = day_start_hour + 24

        # Get anomaly flags for this day's hours
        day_data = hourly_df.iloc[day_start_hour:day_end_hour]
        if len(day_data) > 0 and "is_anomaly" in day_data.columns:
            daily_ground_truth[day_offset] = day_data["is_anomaly"].sum() > 0

    return daily_ground_truth


def compute_confusion_matrix(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict[str, int]:
    """
    Compute confusion matrix metrics.

    Args:
        predictions: Binary predictions (True = anomaly)
        ground_truth: Binary ground truth (True = anomaly)

    Returns:
        Dict with TP, FP, TN, FN counts
    """
    tp = int(np.sum(predictions & ground_truth))
    fp = int(np.sum(predictions & ~ground_truth))
    tn = int(np.sum(~predictions & ~ground_truth))
    fn = int(np.sum(~predictions & ground_truth))

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def aggregate_to_daily(
    point_scores: np.ndarray,
    timestamps: np.ndarray,
    point_threshold: float,
    k: int = 3,
    hours_per_day: int = 24,
    test_start_hour: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate sliding window point scores to daily decisions.

    With stride=1, a single hour appears in multiple overlapping windows.
    This function computes per-hour scores by averaging across all windows
    that contain each hour, then applies HardCriterion once per calendar day.

    This prevents overlap inflation where a few problematic hours cause
    many overlapping windows to be flagged.

    Args:
        point_scores: Per-point NLL scores, shape (num_windows, window_size)
        timestamps: Starting hour index for each window, shape (num_windows,)
        point_threshold: Point-level threshold for anomaly detection
        k: HardCriterion k - number of anomalous points to flag a day
        hours_per_day: Hours per day (24)
        test_start_hour: First hour in the test period

    Returns:
        Tuple of:
            - daily_predictions: (num_days,) binary array, True = anomalous day
            - daily_point_scores: (num_days, 24) average per-hour scores
            - daily_anomaly_counts: (num_days,) count of anomalous hours per day
    """
    if len(point_scores) == 0 or len(timestamps) == 0:
        return np.array([]), np.array([]), np.array([])

    window_size = point_scores.shape[1]

    # Determine the range of hours covered
    min_hour = int(timestamps.min())
    max_hour = int(timestamps.max()) + window_size - 1

    # Compute per-hour scores by averaging across all windows containing each hour
    hour_scores = {}  # hour_idx -> list of scores
    for window_idx, start_hour in enumerate(timestamps):
        for offset in range(window_size):
            hour_idx = int(start_hour) + offset
            if hour_idx not in hour_scores:
                hour_scores[hour_idx] = []
            hour_scores[hour_idx].append(point_scores[window_idx, offset])

    # Average scores per hour
    hour_avg_scores = {h: np.mean(scores) for h, scores in hour_scores.items()}

    # Group by calendar day
    first_test_day = min_hour // hours_per_day
    last_test_day = max_hour // hours_per_day
    num_days = last_test_day - first_test_day + 1

    daily_point_scores = np.full((num_days, hours_per_day), np.nan)
    daily_predictions = np.zeros(num_days, dtype=bool)
    daily_anomaly_counts = np.zeros(num_days, dtype=int)

    for day_offset in range(num_days):
        day_idx = first_test_day + day_offset
        day_start_hour = day_idx * hours_per_day

        # Collect scores for each hour of this day
        for hour_of_day in range(hours_per_day):
            hour_idx = day_start_hour + hour_of_day
            if hour_idx in hour_avg_scores:
                daily_point_scores[day_offset, hour_of_day] = hour_avg_scores[hour_idx]

        # Apply HardCriterion for this day
        valid_scores = daily_point_scores[day_offset, ~np.isnan(daily_point_scores[day_offset])]
        if len(valid_scores) > 0:
            n_anomalous = np.sum(valid_scores < point_threshold)
            daily_anomaly_counts[day_offset] = n_anomalous
            daily_predictions[day_offset] = n_anomalous >= k

    return daily_predictions, daily_point_scores, daily_anomaly_counts


def evaluate_combo(
    registry: "FCVAERegistry",
    preprocessor: "TransactionPreprocessor",
    combo: Tuple[str, str],
    output_path: Path,
    device: torch.device,
    stride: int = 1,
    batch_size: int = 64,
    no_plots: bool = False,
) -> Dict:
    """
    Evaluate a single combo and generate visualizations.

    Args:
        registry: FCVAERegistry with trained models
        preprocessor: TransactionPreprocessor with data
        combo: (network_type, txn_type) tuple
        output_path: Directory to save plots
        device: Torch device
        stride: Sliding window stride
        batch_size: Batch size for scoring

    Returns:
        Dict with evaluation results
    """
    from torch.utils.data import DataLoader
    from app.transaction_preprocessor import SlidingWindowDataset

    combo_name = f"{combo[0]}/{combo[1]}"
    logger.info(f"\nEvaluating {combo_name}...")

    model = registry.get_model(combo)
    scorer = registry.get_scorer(combo)

    # Get test windows from preprocessor
    splits = preprocessor.create_sliding_splits(combo=combo, window_size=24, stride=stride)
    normalized = preprocessor.normalize_sliding_windows(combo=combo, splits=splits, fit_on="train")

    test_data = normalized.get("test")
    if test_data is None or len(test_data[0]) == 0:
        logger.warning(f"No test data for {combo_name}")
        return {}

    # Unpack tuple (windows, labels) returned by normalize_sliding_windows
    test_windows, test_labels = test_data
    test_dataset = SlidingWindowDataset(test_windows, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Score test data
    point_scores, window_scores = scorer.score_batch(model, test_loader, device)

    # Generate plots (unless disabled)
    output_path.mkdir(parents=True, exist_ok=True)

    if not no_plots:
        # 1. Reconstruction plot (first 5 windows)
        plot_reconstruction(
            model=model,
            windows=test_windows,
            scaler=registry.get_scaler((combo[0], combo[1] + "_sliding")) or registry.get_scaler(combo),
            device=device,
            output_path=output_path,
            combo_name=combo_name,
            num_windows=5,
        )

        # 2. Score distributions
        plot_score_distributions(
            point_scores=point_scores,
            window_scores=window_scores,
            point_threshold=scorer.point_threshold,
            window_threshold=scorer.window_threshold,
            output_path=output_path,
            combo_name=combo_name,
        )

        # 3. Score heatmap
        plot_score_heatmap(
            point_scores=point_scores,
            point_threshold=scorer.point_threshold,
            output_path=output_path,
            combo_name=combo_name,
        )

        # 4. Training history
        if combo in registry.training_histories:
            plot_training_history(
                history=registry.training_histories[combo],
                output_path=output_path,
                combo_name=combo_name,
            )

    # Compute summary metrics - both per-sliding-window and per-day
    point_predictions = point_scores < scorer.point_threshold
    window_predictions = scorer.predict_windows_from_points(point_predictions)

    # Get timestamps for test windows to enable daily aggregation
    sw_data = preprocessor.combo_sliding_windows.get(combo, {})
    all_timestamps = sw_data.get("timestamps", np.array([]))

    # Filter timestamps for test windows only (same indexing as test_windows)
    # Test windows are those with starting hour >= threshold_end_hour
    hours_per_day = 24
    train_days = preprocessor.config.train_days
    val_days = preprocessor.config.val_days
    threshold_days = preprocessor.config.threshold_days
    test_start_hour = (train_days + val_days + threshold_days) * hours_per_day

    test_timestamps = all_timestamps[all_timestamps >= test_start_hour]

    # Aggregate to daily predictions (prevents overlap inflation)
    daily_predictions, daily_point_scores, daily_anomaly_counts = aggregate_to_daily(
        point_scores=point_scores,
        timestamps=test_timestamps,
        point_threshold=scorer.point_threshold,
        k=scorer.config.hard_criterion_k,
        hours_per_day=hours_per_day,
        test_start_hour=test_start_hour,
    )

    # Get ground truth labels if available
    test_start_day = train_days + val_days + threshold_days
    num_test_days = len(daily_predictions)
    daily_ground_truth = get_daily_ground_truth(
        preprocessor=preprocessor,
        combo=combo,
        test_start_day=test_start_day,
        num_test_days=num_test_days,
    )

    # Compute confusion matrix
    confusion = compute_confusion_matrix(daily_predictions, daily_ground_truth)
    has_ground_truth = daily_ground_truth.sum() > 0 or num_test_days > 5  # Extended data

    results = {
        # Sliding window results (for backward compatibility)
        "num_windows": len(window_scores),
        "num_predicted_anomalies_windows": int(window_predictions.sum()),
        "mean_window_score": float(window_scores.mean()),
        "point_threshold": float(scorer.point_threshold),
        "window_threshold": float(scorer.window_threshold) if scorer.window_threshold else None,
        "point_scores": point_scores,
        "window_scores": window_scores,
        # Daily aggregated results (more interpretable)
        "num_days": len(daily_predictions),
        "num_predicted_anomalies_daily": int(daily_predictions.sum()),
        "daily_predictions": daily_predictions,
        "daily_point_scores": daily_point_scores,
        "daily_anomaly_counts": daily_anomaly_counts,
        # Ground truth comparison
        "daily_ground_truth": daily_ground_truth,
        "confusion_matrix": confusion,
        "has_ground_truth": has_ground_truth,
    }

    logger.info(f"  Sliding windows: {results['num_windows']}")
    logger.info(f"  Window-level anomalies: {results['num_predicted_anomalies_windows']} (overlapping)")
    logger.info(f"  Test days: {results['num_days']}")
    logger.info(f"  Daily-aggregated anomalies: {results['num_predicted_anomalies_daily']}")
    logger.info(f"  Point threshold: {results['point_threshold']:.4f}")

    if has_ground_truth:
        logger.info(f"  Ground truth anomalies: {int(daily_ground_truth.sum())}")
        logger.info(f"  Confusion: TP={confusion['TP']}, FP={confusion['FP']}, "
                    f"TN={confusion['TN']}, FN={confusion['FN']}")

    return results


def main():
    """Main evaluation script for FCVAE models."""
    parser = argparse.ArgumentParser(
        description="Evaluate FCVAE models and generate visualizations"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/transactions_fcvae",
        help="Directory containing saved FCVAE models"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/synthetic_transactions.csv",
        help="Path to synthetic transactions CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/fcvae",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--combo",
        type=str,
        default="all",
        help="Specific combo (e.g., 'Accel/CMP') or 'all'"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for scoring"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (evaluation metrics only)"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not HAS_MATPLOTLIB and not args.no_plots:
        logger.error("matplotlib is required for visualization. Install with: pip install matplotlib")
        return

    print("\n" + "=" * 60)
    print("FCVAE MODEL EVALUATION")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load registry
    from app.fcvae_registry import FCVAERegistry
    from app.transaction_config import COMBO_KEYS, TransactionPreprocessorConfig
    from app.transaction_preprocessor import TransactionPreprocessor

    registry = FCVAERegistry(device=device)
    registry.load_all(args.model_dir)
    print(f"Loaded models from: {args.model_dir}")

    # Load preprocessor
    preprocessor = TransactionPreprocessor(config=TransactionPreprocessorConfig())
    preprocessor.load_and_aggregate(args.data_path)
    print(f"Loaded data from: {args.data_path}")

    # Output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Determine combos to evaluate
    if args.combo.lower() == "all":
        combos_to_eval = list(COMBO_KEYS)
    else:
        # Parse combo string like "Accel/CMP"
        parts = args.combo.split("/")
        if len(parts) == 2:
            combos_to_eval = [(parts[0], parts[1])]
        else:
            logger.error(f"Invalid combo format: {args.combo}. Use 'Accel/CMP' format.")
            return

    # Evaluate each combo
    all_results = {}
    for combo in combos_to_eval:
        if combo not in registry.models:
            logger.warning(f"No model found for {combo}, skipping")
            continue

        results = evaluate_combo(
            registry=registry,
            preprocessor=preprocessor,
            combo=combo,
            output_path=output_path,
            device=device,
            stride=args.stride,
            batch_size=args.batch_size,
            no_plots=args.no_plots,
        )

        if results:
            all_results[combo] = results

    # Generate combo comparison plot
    if len(all_results) > 1 and not args.no_plots:
        plot_combo_comparison(
            registry=registry,
            all_scores=all_results,
            output_path=output_path,
        )

    # Generate test anomaly reconstruction plots for each combo
    if not args.no_plots:
        for combo in combos_to_eval:
            if combo in registry.models:
                plot_test_anomalies(
                    registry=registry,
                    preprocessor=preprocessor,
                    combo=combo,
                    output_path=output_path,
                    device=device,
                    test_start_day=preprocessor.config.train_days + preprocessor.config.val_days + preprocessor.config.threshold_days,
                )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Check if we have ground truth labels
    any_has_ground_truth = any(r.get("has_ground_truth", False) for r in all_results.values())

    if any_has_ground_truth:
        # Show confusion matrix results
        print(f"\n{'Combo':<20} {'Days':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'Mean Score':>12}")
        print("-" * 65)

        for combo, results in all_results.items():
            combo_name = f"{combo[0]}/{combo[1]}"
            cm = results.get("confusion_matrix", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
            print(f"{combo_name:<20} {results['num_days']:>6} "
                  f"{cm['TP']:>4} {cm['FP']:>4} {cm['TN']:>4} {cm['FN']:>4} "
                  f"{results['mean_window_score']:>12.4f}")

        # Show per-day breakdown for first combo
        first_combo = list(all_results.keys())[0]
        first_results = all_results[first_combo]
        print(f"\nPer-day breakdown for {first_combo[0]}/{first_combo[1]}:")
        print(f"{'Day':>4} {'Anom Hrs':>10} {'Predicted':>12} {'Ground Truth':>14}")
        print("-" * 45)

        for i in range(first_results["num_days"]):
            pred = "ANOMALY" if first_results["daily_predictions"][i] else "normal"
            gt = "ANOMALY" if first_results["daily_ground_truth"][i] else "normal"
            anom_hrs = first_results["daily_anomaly_counts"][i]
            match = "✓" if first_results["daily_predictions"][i] == first_results["daily_ground_truth"][i] else "✗"
            print(f"{i:>4} {anom_hrs:>10} {pred:>12} {gt:>14}  {match}")

    else:
        # Original output without ground truth
        print(f"\n{'Combo':<20} {'Test Days':>10} {'Daily Pred':>12} {'Window Pred':>12} {'Mean Score':>12}")
        print("-" * 70)

        for combo, results in all_results.items():
            combo_name = f"{combo[0]}/{combo[1]}"
            print(f"{combo_name:<20} {results['num_days']:>10} "
                  f"{results['num_predicted_anomalies_daily']:>12} "
                  f"{results['num_predicted_anomalies_windows']:>12} "
                  f"{results['mean_window_score']:>12.4f}")

        print("\nNote: 'Daily Pred' = daily-aggregated predictions (overlap-adjusted)")
        print("      'Window Pred' = raw overlapping window predictions (inflated)")
        print("      No ground truth labels available - all predictions are assumed false positives")

    print("\n" + "=" * 60)
    if args.no_plots:
        print("Plots were skipped (--no-plots flag)")
    else:
        print(f"Plots saved to: {output_path}")
    print("=" * 60)


def plot_test_anomalies(
    registry: "FCVAERegistry",
    preprocessor: "TransactionPreprocessor",
    combo: Tuple[str, str],
    output_path: Path,
    device: torch.device,
    test_start_day: int = 25,
) -> None:
    """
    Plot reconstruction for test days with injected anomalies.

    Finds days in the test period that have is_anomaly=1 and generates
    reconstruction plots showing how the model scores those anomalies.

    Args:
        registry: FCVAERegistry with trained models
        preprocessor: TransactionPreprocessor with data
        combo: (network_type, txn_type) tuple
        output_path: Directory to save plots
        device: Torch device
        test_start_day: First day of test period (default 25)
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping test anomaly plots")
        return

    combo_name = f"{combo[0]}/{combo[1]}"
    logger.info(f"Generating test anomaly plots for {combo_name}...")

    model = registry.get_model(combo)
    scorer = registry.get_scorer(combo)

    # Get hourly data with anomaly labels
    hourly_df = preprocessor.combo_hourly.get(combo)
    if hourly_df is None or "is_anomaly" not in hourly_df.columns:
        logger.warning(f"No anomaly labels found for {combo_name}")
        return

    # Get scaler
    scaler_key = (combo[0], combo[1] + "_sliding")
    scaler = registry.get_scaler(scaler_key) or registry.get_scaler(combo)
    if scaler is None:
        logger.warning(f"No scaler found for {combo_name}")
        return

    # Find days with anomalies in test period
    total_days = len(hourly_df) // 24
    anomaly_days = []

    for day_idx in range(test_start_day, total_days):
        day_start = day_idx * 24
        day_end = day_start + 24
        day_data = hourly_df.iloc[day_start:day_end]

        if day_data["is_anomaly"].sum() > 0:
            anomaly_hours = day_data[day_data["is_anomaly"] == 1].index.tolist()
            anomaly_hours_of_day = [h - day_start for h in anomaly_hours]
            anomaly_days.append({
                "day_idx": day_idx,
                "test_day": day_idx - test_start_day,
                "anomaly_hours": anomaly_hours_of_day,
                "count": day_data["is_anomaly"].sum(),
            })

    if not anomaly_days:
        logger.warning(f"No anomaly days found in test period for {combo_name}")
        return

    logger.info(f"  Found {len(anomaly_days)} days with anomalies in test period")

    # For each anomaly day, create a reconstruction plot
    for anom_info in anomaly_days:
        day_idx = anom_info["day_idx"]
        test_day = anom_info["test_day"]
        anomaly_hours = anom_info["anomaly_hours"]

        # Extract the 24-hour window for this day
        day_start = day_idx * 24
        day_end = day_start + 24
        window_raw = hourly_df.iloc[day_start:day_end]["count"].values.astype(np.float32)

        # Normalize
        window_norm = scaler.transform(window_raw.reshape(-1, 1)).flatten()

        # Create anomaly labels array
        anomaly_labels = np.zeros(24, dtype=np.float32)
        for h in anomaly_hours:
            if 0 <= h < 24:
                anomaly_labels[h] = 1.0

        # Find a normal day for comparison (use test_start_day)
        normal_day_idx = test_start_day
        normal_start = normal_day_idx * 24
        normal_end = normal_start + 24
        normal_raw = hourly_df.iloc[normal_start:normal_end]["count"].values.astype(np.float32)
        normal_norm = scaler.transform(normal_raw.reshape(-1, 1)).flatten()

        # Determine anomaly type based on whether anomaly hours have higher or lower counts
        anomaly_hour_counts = window_raw[anomaly_labels > 0]
        normal_hour_counts = window_raw[anomaly_labels == 0]

        if len(anomaly_hour_counts) > 0 and len(normal_hour_counts) > 0:
            if np.mean(anomaly_hour_counts) > np.mean(normal_hour_counts) * 1.5:
                anomaly_type = "spike"
            elif np.mean(anomaly_hour_counts) < np.mean(normal_hour_counts) * 0.5:
                anomaly_type = "outage"
            else:
                anomaly_type = "anomaly"
        else:
            anomaly_type = "anomaly"

        # Use the existing plot function
        plot_anomaly_reconstruction(
            model=model,
            normal_window=normal_norm,
            anomaly_window=window_norm,
            anomaly_labels=anomaly_labels,
            point_threshold=scorer.point_threshold,
            device=device,
            output_path=output_path,
            combo_name=f"{combo[0]}_{combo[1]}_test_day{test_day}",
            anomaly_type=anomaly_type,
        )

        logger.info(f"  Plotted test day {test_day} ({anomaly_type}, {len(anomaly_hours)} anomalous hours)")


def plot_val_anomalies(
    registry: "FCVAERegistry",
    preprocessor: "TransactionPreprocessor",
    combo: Tuple[str, str],
    output_path: Path,
    device: torch.device,
) -> None:
    """
    Plot reconstruction for validation days with injected anomalies.

    These are the anomalies used for F1-based threshold calibration.

    Args:
        registry: FCVAERegistry with trained models
        preprocessor: TransactionPreprocessor with data
        combo: (network_type, txn_type) tuple
        output_path: Directory to save plots
        device: Torch device
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping validation anomaly plots")
        return

    combo_name = f"{combo[0]}/{combo[1]}"
    logger.info(f"Generating validation anomaly plots for {combo_name}...")

    model = registry.get_model(combo)
    scorer = registry.get_scorer(combo)

    # Get hourly data with anomaly labels
    hourly_df = preprocessor.combo_hourly.get(combo)
    if hourly_df is None or "is_anomaly" not in hourly_df.columns:
        logger.warning(f"No anomaly labels found for {combo_name}")
        return

    # Get scaler
    scaler_key = (combo[0], combo[1] + "_sliding")
    scaler = registry.get_scaler(scaler_key) or registry.get_scaler(combo)
    if scaler is None:
        logger.warning(f"No scaler found for {combo_name}")
        return

    # Determine validation period
    # For CSV-based splits, we need to find hours marked as "val"
    if preprocessor.has_csv_splits and combo in preprocessor.combo_hour_splits:
        hourly_splits = preprocessor.combo_hour_splits[combo].values
        val_mask = hourly_splits == "val"
        val_start_hour = int(np.argmax(val_mask))
        val_end_hour = len(hourly_splits) - int(np.argmax(val_mask[::-1]))
    else:
        train_days = preprocessor.config.train_days
        val_days = preprocessor.config.val_days
        val_start_hour = train_days * 24
        val_end_hour = (train_days + val_days) * 24

    val_start_day = val_start_hour // 24
    val_end_day = val_end_hour // 24

    # Find days with anomalies in validation period
    anomaly_days = []
    for day_idx in range(val_start_day, val_end_day):
        day_start = day_idx * 24
        day_end = day_start + 24
        if day_end > len(hourly_df):
            break

        day_data = hourly_df.iloc[day_start:day_end]

        if "is_anomaly" in day_data.columns and day_data["is_anomaly"].sum() > 0:
            anomaly_hours_of_day = list(np.where(day_data["is_anomaly"].values == 1)[0])
            anomaly_days.append({
                "day_idx": day_idx,
                "val_day": day_idx - val_start_day,
                "anomaly_hours": anomaly_hours_of_day,
                "count": int(day_data["is_anomaly"].sum()),
            })

    if not anomaly_days:
        logger.warning(f"No anomaly days found in validation period for {combo_name}")
        return

    logger.info(f"  Found {len(anomaly_days)} days with anomalies in validation period")

    # Create validation subdirectory
    val_output_path = output_path / "validation_anomalies"
    val_output_path.mkdir(parents=True, exist_ok=True)

    # For each anomaly day, create a reconstruction plot
    for anom_info in anomaly_days:
        day_idx = anom_info["day_idx"]
        val_day = anom_info["val_day"]
        anomaly_hours = anom_info["anomaly_hours"]

        # Extract the 24-hour window for this day
        day_start = day_idx * 24
        day_end = day_start + 24
        window_raw = hourly_df.iloc[day_start:day_end]["count"].values.astype(np.float32)

        # Normalize
        window_norm = scaler.transform(window_raw.reshape(-1, 1)).flatten()

        # Create anomaly labels array
        anomaly_labels = np.zeros(24, dtype=np.float32)
        for h in anomaly_hours:
            if 0 <= h < 24:
                anomaly_labels[h] = 1.0

        # Find a normal day for comparison (use first train day)
        normal_day_idx = 0
        normal_start = normal_day_idx * 24
        normal_end = normal_start + 24
        normal_raw = hourly_df.iloc[normal_start:normal_end]["count"].values.astype(np.float32)
        normal_norm = scaler.transform(normal_raw.reshape(-1, 1)).flatten()

        # Determine anomaly type
        anomaly_hour_counts = window_raw[anomaly_labels > 0]
        normal_hour_counts = window_raw[anomaly_labels == 0]

        if len(anomaly_hour_counts) > 0 and len(normal_hour_counts) > 0:
            if np.mean(anomaly_hour_counts) > np.mean(normal_hour_counts) * 1.5:
                anomaly_type = "spike"
            elif np.mean(anomaly_hour_counts) < np.mean(normal_hour_counts) * 0.5:
                anomaly_type = "dip"
            else:
                anomaly_type = "anomaly"
        else:
            anomaly_type = "anomaly"

        # Use the existing plot function
        plot_anomaly_reconstruction(
            model=model,
            normal_window=normal_norm,
            anomaly_window=window_norm,
            anomaly_labels=anomaly_labels,
            point_threshold=scorer.point_threshold,
            device=device,
            output_path=val_output_path,
            combo_name=f"{combo[0]}_{combo[1]}_val_day{val_day}",
            anomaly_type=anomaly_type,
        )

        logger.info(f"  Plotted validation day {val_day} ({anomaly_type}, {len(anomaly_hours)} anomalous hours)")


def plot_false_positives(
    registry: "FCVAERegistry",
    preprocessor: "TransactionPreprocessor",
    combo: Tuple[str, str],
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    predictions: np.ndarray,
    point_scores: np.ndarray,
    output_path: Path,
    device: torch.device,
) -> List[int]:
    """
    Plot reconstruction for false positive windows.

    A false positive is a window predicted as anomaly but with no actual anomaly labels.

    Args:
        registry: FCVAERegistry with trained models
        preprocessor: TransactionPreprocessor with data
        combo: (network_type, txn_type) tuple
        test_windows: Normalized test windows (N, 24)
        test_labels: Per-point anomaly labels (N, 24)
        predictions: Binary window predictions (N,)
        point_scores: Per-point NLL scores (N, 24)
        output_path: Directory to save plots
        device: Torch device

    Returns:
        List of false positive window indices
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping false positive plots")
        return []

    combo_name = f"{combo[0]}/{combo[1]}"
    model = registry.get_model(combo)
    scorer = registry.get_scorer(combo)

    # Find false positives: predicted anomaly but no actual anomaly labels in window
    window_labels = (test_labels.sum(axis=1) > 0).astype(int)
    fp_mask = (predictions == 1) & (window_labels == 0)
    fp_indices = np.where(fp_mask)[0]

    if len(fp_indices) == 0:
        logger.info(f"  No false positives found for {combo_name}")
        return []

    logger.info(f"  Found {len(fp_indices)} false positive windows for {combo_name}")

    # Create FP subdirectory
    fp_output_path = output_path / "false_positives"
    fp_output_path.mkdir(parents=True, exist_ok=True)

    # Plot each false positive
    for fp_idx in fp_indices:
        window_norm = test_windows[fp_idx]
        window_scores = point_scores[fp_idx]

        # Create detailed FP plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        hours = np.arange(24)

        # Top: Reconstruction with confidence bands
        ax = axes[0]
        x = torch.FloatTensor(window_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            mu_x, var_x = model.reconstruct(x)
            mu_x = mu_x.squeeze().cpu().numpy()
            var_x = var_x.squeeze().cpu().numpy()
        std_x = np.sqrt(var_x)

        ax.plot(hours, window_norm, 'b-', linewidth=2, label='Original', marker='o', markersize=4)
        ax.plot(hours, mu_x, 'r--', linewidth=2, label='Reconstruction (μ_x)')
        ax.fill_between(hours, mu_x - 2 * std_x, mu_x + 2 * std_x,
                        alpha=0.3, color='red', label='±2σ_x confidence')

        # Highlight hours that exceeded threshold
        anomalous_hours = window_scores < scorer.point_threshold
        if np.any(anomalous_hours):
            ax.scatter(hours[anomalous_hours], window_norm[anomalous_hours],
                       c='orange', s=100, zorder=5, marker='s', label='Below threshold')

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Normalized Count')
        ax.set_title(f'{combo_name} - False Positive Window #{fp_idx}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 2))

        # Bottom: Per-point NLL scores
        ax = axes[1]
        colors = ['orange' if s < scorer.point_threshold else 'steelblue' for s in window_scores]
        bars = ax.bar(hours, window_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(scorer.point_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {scorer.point_threshold:.2f}')

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('NLL Score (lower = more anomalous)')
        ax.set_title(f'Point NLL Scores - {np.sum(anomalous_hours)}/24 below threshold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 2))

        plt.tight_layout()
        save_path = fp_output_path / f"{combo[0]}_{combo[1]}_fp_window{fp_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"    Saved FP plot: {save_path.name}")

    return list(fp_indices)


def evaluate_with_detailed_plots(
    registry: "FCVAERegistry",
    preprocessor: "TransactionPreprocessor",
    combo: Tuple[str, str],
    output_path: Path,
    device: torch.device,
    stride: int = 1,
    batch_size: int = 64,
    no_plots: bool = False,
) -> Dict:
    """
    Comprehensive evaluation with detailed anomaly and FP plots.

    Generates:
    1. Standard evaluation plots (reconstruction, score distributions, heatmaps)
    2. Validation anomaly reconstruction plots (threshold calibration analysis)
    3. Test anomaly reconstruction plots
    4. False positive analysis plots

    Args:
        registry: FCVAERegistry with trained models
        preprocessor: TransactionPreprocessor with data
        combo: (network_type, txn_type) tuple
        output_path: Directory to save plots
        device: Torch device
        stride: Sliding window stride
        batch_size: Batch size for scoring

    Returns:
        Dict with evaluation results including FP indices
    """
    from torch.utils.data import DataLoader
    from app.transaction_preprocessor import SlidingWindowDataset

    combo_name = f"{combo[0]}/{combo[1]}"
    logger.info(f"\nComprehensive evaluation for {combo_name}...")

    model = registry.get_model(combo)
    scorer = registry.get_scorer(combo)

    # Get test windows from preprocessor
    splits = preprocessor.create_sliding_splits(combo=combo, window_size=24, stride=stride)
    normalized = preprocessor.normalize_sliding_windows(combo=combo, splits=splits, fit_on="train")

    test_data = normalized.get("test")
    val_data = normalized.get("val")

    if test_data is None or len(test_data[0]) == 0:
        logger.warning(f"No test data for {combo_name}")
        return {}

    test_windows, test_labels = test_data
    test_dataset = SlidingWindowDataset(test_windows, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Score test data
    point_scores, window_scores = scorer.score_batch(model, test_loader, device)

    # Get predictions
    point_predictions = point_scores < scorer.point_threshold
    window_predictions = scorer.predict_windows_from_points(point_predictions)

    # Create output subdirectories
    output_path.mkdir(parents=True, exist_ok=True)
    test_anomaly_path = output_path / "test_anomalies"
    test_anomaly_path.mkdir(parents=True, exist_ok=True)

    fp_indices = []

    if not no_plots:
        # 1. Standard plots
        plot_reconstruction(
            model=model,
            windows=test_windows,
            scaler=registry.get_scaler((combo[0], combo[1] + "_sliding")) or registry.get_scaler(combo),
            device=device,
            output_path=output_path,
            combo_name=combo_name,
            num_windows=5,
        )

        plot_score_distributions(
            point_scores=point_scores,
            window_scores=window_scores,
            point_threshold=scorer.point_threshold,
            window_threshold=scorer.window_threshold,
            output_path=output_path,
            combo_name=combo_name,
        )

        plot_score_heatmap(
            point_scores=point_scores,
            point_threshold=scorer.point_threshold,
            output_path=output_path,
            combo_name=combo_name,
        )

        if combo in registry.training_histories:
            plot_training_history(
                history=registry.training_histories[combo],
                output_path=output_path,
                combo_name=combo_name,
            )

        # 2. Validation anomaly plots (for threshold calibration analysis)
        plot_val_anomalies(
            registry=registry,
            preprocessor=preprocessor,
            combo=combo,
            output_path=output_path,
            device=device,
        )

        # 3. Test anomaly plots
        # Use CSV splits if available
        if preprocessor.has_csv_splits:
            hourly_splits = preprocessor.combo_hour_splits[combo].values
            test_mask = hourly_splits == "test"
            test_start_hour = int(np.argmax(test_mask))
            test_start_day = test_start_hour // 24
        else:
            test_start_day = (preprocessor.config.train_days +
                              preprocessor.config.val_days +
                              preprocessor.config.threshold_days)

        plot_test_anomalies(
            registry=registry,
            preprocessor=preprocessor,
            combo=combo,
            output_path=test_anomaly_path,
            device=device,
            test_start_day=test_start_day,
        )

        # 4. False positive analysis
        fp_indices = plot_false_positives(
            registry=registry,
            preprocessor=preprocessor,
            combo=combo,
            test_windows=test_windows,
            test_labels=test_labels,
            predictions=window_predictions,
            point_scores=point_scores,
            output_path=output_path,
            device=device,
        )

    # Compute metrics
    window_labels = (test_labels.sum(axis=1) > 0).astype(int)
    tp = int(((window_predictions == 1) & (window_labels == 1)).sum())
    fp = int(((window_predictions == 1) & (window_labels == 0)).sum())
    fn = int(((window_predictions == 0) & (window_labels == 1)).sum())
    tn = int(((window_predictions == 0) & (window_labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "num_windows": len(window_scores),
        "num_actual_anomalies": int(window_labels.sum()),
        "num_predicted_anomalies": int(window_predictions.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_indices": fp_indices,
        "point_threshold": float(scorer.point_threshold),
        "window_threshold": float(scorer.window_threshold) if scorer.window_threshold else None,
        "point_scores": point_scores,
        "window_scores": window_scores,
    }

    logger.info(f"  Windows: {results['num_windows']}, TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    logger.info(f"  P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return results


def run_detailed_evaluation():
    """Run detailed evaluation with all anomaly visualizations."""
    parser = argparse.ArgumentParser(
        description="Detailed FCVAE evaluation with anomaly and FP visualizations"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/transactions_fcvae_60",
        help="Directory containing saved FCVAE models"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/synthetic_transactions_v2_split60.csv",
        help="Path to synthetic transactions CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/fcvae_60",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for scoring"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (evaluation metrics only)"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not HAS_MATPLOTLIB and not args.no_plots:
        logger.error("matplotlib is required. Install with: pip install matplotlib")
        return

    print("\n" + "=" * 70)
    print("FCVAE DETAILED EVALUATION")
    print("With Validation Anomalies, Test Anomalies, and False Positive Analysis")
    print("=" * 70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load registry
    from app.fcvae_registry import FCVAERegistry
    from app.transaction_config import COMBO_KEYS, TransactionPreprocessorConfig
    from app.transaction_preprocessor import TransactionPreprocessor

    registry = FCVAERegistry(device=device)
    registry.load_all(args.model_dir)
    print(f"Loaded models from: {args.model_dir}")

    # Load preprocessor
    preprocessor = TransactionPreprocessor(config=TransactionPreprocessorConfig())
    preprocessor.load_and_aggregate(args.data_path)
    print(f"Loaded data from: {args.data_path}")

    # Output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Evaluate each combo
    all_results = {}
    for combo in COMBO_KEYS:
        if combo not in registry.models:
            logger.warning(f"No model found for {combo}, skipping")
            continue

        results = evaluate_with_detailed_plots(
            registry=registry,
            preprocessor=preprocessor,
            combo=combo,
            output_path=output_path,
            device=device,
            stride=args.stride,
            batch_size=args.batch_size,
            no_plots=args.no_plots,
        )

        if results:
            all_results[combo] = results

    # Generate combo comparison
    if len(all_results) > 1 and not args.no_plots:
        plot_combo_comparison(
            registry=registry,
            all_scores=all_results,
            output_path=output_path,
        )

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Combo':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 86)

    for combo, results in all_results.items():
        combo_name = f"{combo[0]}/{combo[1]}"
        print(f"{combo_name:<20} {results['tp']:>6} {results['fp']:>6} "
              f"{results['fn']:>6} {results['tn']:>6} "
              f"{results['precision']:>10.4f} {results['recall']:>10.4f} {results['f1']:>10.4f}")

    if not args.no_plots:
        print("\n" + "-" * 70)
        print("Output Structure:")
        print(f"  {output_path}/")
        print(f"    ├── [combo]_reconstruction.png      # Normal test windows")
        print(f"    ├── [combo]_score_distribution.png  # Score histograms")
        print(f"    ├── [combo]_score_heatmap.png       # Per-hour score heatmap")
        print(f"    ├── [combo]_training_history.png    # Loss curves")
        print(f"    ├── validation_anomalies/           # Threshold calibration anomalies")
        print(f"    │   └── [combo]_val_day*_anomaly_*.png")
        print(f"    ├── test_anomalies/                 # Test set injected anomalies")
        print(f"    │   └── [combo]_test_day*_anomaly_*.png")
        print(f"    └── false_positives/                # FP window analysis")
        print(f"        └── [combo]_fp_window*.png")
    else:
        print("\n" + "-" * 70)
        print("Plots were skipped (--no-plots flag)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "detailed":
        sys.argv.pop(1)
        run_detailed_evaluation()
    else:
        main()
