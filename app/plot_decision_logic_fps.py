"""
Plot False Positives for a Specific Decision Mode

Generates reconstruction plots for false positive windows using a specified
decision logic (e.g., severity_0.5) to understand why normal windows are flagged.

Usage:
    python -m app.plot_decision_logic_fps \
        --model-dir models/fcvae_60_kl \
        --data-path data/synthetic_transactions_v2_split60.csv \
        --output-dir results/fcvae_60_kl/false_positives_severity_0.5 \
        --decision-mode severity_0.5 \
        --max-per-combo 3
"""

import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)

# Decision mode configurations
DECISION_MODES = {
    "count_k3": {"mode": "count_only", "k": 3},
    "k1": {"mode": "k1", "k": 1},
    "severity_0.3": {"mode": "severity", "severity_margin": 0.3, "k": 3},
    "severity_0.5": {"mode": "severity", "severity_margin": 0.5, "k": 3},
    "zscore_2.5": {"mode": "zscore", "outlier_z_threshold": 2.5, "k": 3},
    "zscore_3.0": {"mode": "zscore", "outlier_z_threshold": 3.0, "k": 3},
    "hybrid": {"mode": "hybrid", "severity_margin": 0.5, "outlier_z_threshold": 3.0, "k": 3},
}


def apply_decision_mode(scorer, mode_config: Dict) -> None:
    """Apply decision mode configuration to scorer."""
    scorer.config.decision_mode = mode_config["mode"]
    if "severity_margin" in mode_config:
        scorer.config.severity_margin = mode_config["severity_margin"]
    if "outlier_z_threshold" in mode_config:
        scorer.config.outlier_z_threshold = mode_config["outlier_z_threshold"]
    if "k" in mode_config:
        scorer.config.hard_criterion_k = mode_config["k"]


def plot_fp_window(
    model,
    scorer,
    window_norm: np.ndarray,
    window_scores: np.ndarray,
    combo: Tuple[str, str],
    fp_idx: int,
    output_path: Path,
    device: torch.device,
    mode_name: str,
) -> None:
    """
    Generate a detailed false positive plot showing reconstruction and scores.

    Args:
        model: FCVAE model
        scorer: FCVAEScorer with threshold
        window_norm: Normalized window data (24,)
        window_scores: Per-point NLL scores (24,)
        combo: (network_type, txn_type) tuple
        fp_idx: Window index (for naming)
        output_path: Directory to save plot
        device: Torch device
        mode_name: Decision mode name (for title)
    """
    combo_name = f"{combo[0]}/{combo[1]}"

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

    # Highlight hours that contributed to detection
    point_threshold = scorer.point_threshold
    anomalous_hours = window_scores < point_threshold

    # For severity mode, also check severe threshold
    severity_margin = getattr(scorer.config, 'severity_margin', 0.5)
    severe_threshold = point_threshold - severity_margin
    severe_hours = window_scores < severe_threshold

    if np.any(severe_hours):
        ax.scatter(hours[severe_hours], window_norm[severe_hours],
                   c='red', s=150, zorder=6, marker='s', label='Below severe threshold')
    if np.any(anomalous_hours & ~severe_hours):
        ax.scatter(hours[anomalous_hours & ~severe_hours], window_norm[anomalous_hours & ~severe_hours],
                   c='orange', s=100, zorder=5, marker='s', label='Below threshold')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Normalized Count')
    ax.set_title(f'{combo_name} - False Positive Window #{fp_idx} ({mode_name})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))

    # Bottom: Per-point NLL scores
    ax = axes[1]
    colors = []
    for s in window_scores:
        if s < severe_threshold:
            colors.append('red')
        elif s < point_threshold:
            colors.append('orange')
        else:
            colors.append('steelblue')

    bars = ax.bar(hours, window_scores, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(point_threshold, color='orange', linestyle='--', linewidth=2,
               label=f'Threshold: {point_threshold:.2f}')
    ax.axhline(severe_threshold, color='red', linestyle=':', linewidth=2,
               label=f'Severe threshold: {severe_threshold:.2f}')

    n_below_threshold = int(np.sum(anomalous_hours))
    n_below_severe = int(np.sum(severe_hours))
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('NLL Score (lower = more anomalous)')
    ax.set_title(f'Point NLL Scores - {n_below_threshold}/24 below threshold, {n_below_severe}/24 below severe')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    save_path = output_path / f"{combo[0]}_{combo[1]}_fp_window{fp_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved FP plot: {save_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot false positives for a specific decision mode"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/fcvae_60_kl",
        help="Directory containing saved FCVAE models"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/synthetic_transactions_v2_split60.csv",
        help="Path to transaction data CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/fcvae_60_kl/false_positives_severity_0.5",
        help="Directory to save FP plots"
    )
    parser.add_argument(
        "--decision-mode",
        type=str,
        default="severity_0.5",
        choices=list(DECISION_MODES.keys()),
        help="Decision mode to use for FP identification"
    )
    parser.add_argument(
        "--max-per-combo",
        type=int,
        default=3,
        help="Maximum FP plots per combo (0 = all)"
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
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for plotting")
        return

    print("\n" + "=" * 80)
    print(f"FCVAE FALSE POSITIVE PLOTTING - {args.decision_mode}")
    print("=" * 80)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load registry
    from app.fcvae_registry import FCVAERegistry
    from app.transaction_config import COMBO_KEYS, TransactionPreprocessorConfig
    from app.transaction_preprocessor import TransactionPreprocessor, SlidingWindowDataset

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

    # Get decision mode config
    mode_config = DECISION_MODES[args.decision_mode]
    print(f"\nDecision mode: {args.decision_mode}")
    print(f"  mode: {mode_config['mode']}")
    if 'severity_margin' in mode_config:
        print(f"  severity_margin: {mode_config['severity_margin']}")
    if 'outlier_z_threshold' in mode_config:
        print(f"  outlier_z_threshold: {mode_config['outlier_z_threshold']}")

    total_fps = 0
    total_plots = 0

    for combo in COMBO_KEYS:
        if combo not in registry.models:
            logger.warning(f"No model found for {combo}, skipping")
            continue

        combo_name = f"{combo[0]}/{combo[1]}"
        print(f"\n{'='*60}")
        print(f"Processing {combo_name}")
        print("=" * 60)

        model = registry.get_model(combo)
        scorer = registry.get_scorer(combo)

        # Apply decision mode
        apply_decision_mode(scorer, mode_config)

        # Get test windows
        splits = preprocessor.create_sliding_splits(combo=combo, window_size=24, stride=args.stride)
        normalized = preprocessor.normalize_sliding_windows(combo=combo, splits=splits, fit_on="train")

        test_data = normalized.get("test")
        if test_data is None or len(test_data[0]) == 0:
            logger.warning(f"No test data for {combo_name}")
            continue

        test_windows, test_labels = test_data
        test_dataset = SlidingWindowDataset(test_windows, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Score test data
        point_scores, window_scores = scorer.score_batch(model, test_loader, device)
        point_predictions = point_scores < scorer.point_threshold

        # Get window predictions with specified decision mode
        window_predictions = scorer.predict_windows_from_points(point_predictions, point_scores)

        # Find false positives: predicted anomaly but no actual anomaly labels
        window_labels = (test_labels.sum(axis=1) > 0).astype(int)
        fp_mask = (window_predictions == 1) & (window_labels == 0)
        fp_indices = np.where(fp_mask)[0]

        print(f"  Test windows: {len(test_windows)}")
        print(f"  Point threshold: {scorer.point_threshold:.4f}")
        print(f"  Window predictions: {window_predictions.sum()} anomalies")
        print(f"  False positives: {len(fp_indices)}")

        total_fps += len(fp_indices)

        if len(fp_indices) == 0:
            print(f"  No false positives found for {combo_name}")
            continue

        # Unique hour analysis
        # Get timestamps for test windows
        sw_data = preprocessor.combo_sliding_windows.get(combo, {})
        all_timestamps = sw_data.get("timestamps", np.array([]))

        # Get test timestamps (filter to test period)
        hours_per_day = 24
        train_days = preprocessor.config.train_days
        val_days = preprocessor.config.val_days
        threshold_days = preprocessor.config.threshold_days
        test_start_hour = (train_days + val_days + threshold_days) * hours_per_day
        test_timestamps = all_timestamps[all_timestamps >= test_start_hour]

        # Ensure we have the right number of timestamps
        if len(test_timestamps) != len(test_windows):
            # Use last N timestamps
            test_timestamps = all_timestamps[-len(test_windows):]

        # Find which absolute hours triggered each FP
        severity_margin = mode_config.get("severity_margin", 0.5)
        severe_threshold = scorer.point_threshold - severity_margin
        triggering_hours = Counter()

        for fp_idx in fp_indices:
            start_hour = int(test_timestamps[fp_idx])
            window_point_scores = point_scores[fp_idx]

            # Find hours that triggered detection based on mode
            mode = mode_config["mode"]
            if mode in ["severity", "hybrid"]:
                # Severity: below threshold OR below severe threshold
                for offset in range(24):
                    if window_point_scores[offset] < severe_threshold:
                        triggering_hours[start_hour + offset] += 1
                    elif window_point_scores[offset] < scorer.point_threshold:
                        triggering_hours[start_hour + offset] += 1
            else:
                # Count-based: just below threshold
                for offset in range(24):
                    if window_point_scores[offset] < scorer.point_threshold:
                        triggering_hours[start_hour + offset] += 1

        # Report unique hours
        unique_hours = len(triggering_hours)
        print(f"\n  Unique Hour Analysis:")
        print(f"    Total FP windows: {len(fp_indices)}")
        print(f"    Unique triggering hours: {unique_hours}")

        # Show most common hours
        if triggering_hours:
            most_common = triggering_hours.most_common(5)
            hours_str = ", ".join([f"hour {h} (in {c} windows)" for h, c in most_common])
            print(f"    Top triggering hours: {hours_str}")

        # Limit plots if requested
        if args.max_per_combo > 0:
            fp_indices_to_plot = fp_indices[:args.max_per_combo]
        else:
            fp_indices_to_plot = fp_indices

        print(f"  Plotting {len(fp_indices_to_plot)} FP windows...")

        # Generate plots
        for fp_idx in fp_indices_to_plot:
            window_norm = test_windows[fp_idx]
            window_point_scores = point_scores[fp_idx]

            plot_fp_window(
                model=model,
                scorer=scorer,
                window_norm=window_norm,
                window_scores=window_point_scores,
                combo=combo,
                fp_idx=fp_idx,
                output_path=output_path,
                device=device,
                mode_name=args.decision_mode,
            )
            total_plots += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total false positives found: {total_fps}")
    print(f"Total plots generated: {total_plots}")
    print(f"Output directory: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
