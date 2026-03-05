"""
Decision Logic Comparison for FCVAE Anomaly Detection

Evaluates different window-level decision logic variants:
1. count_k1: k>=1 points below threshold
2. count_k2: k>=2 points below threshold
3. count_k3: Original - k>=3 points below threshold
4. severity: k>=3 OR any point below (threshold - margin)
5. zscore: k>=3 OR any point with extreme z-score within window
6. hybrid: k>=3 OR severity OR zscore

Usage:
    python -m app.evaluate_decision_logic \
        --model-dir models/fcvae_60_kl \
        --data-path data/synthetic_transactions_v2_split60.csv \
        --output-dir results/fcvae_decision_logic_comparison
"""

import argparse
import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Decision modes to compare
DECISION_MODES = [
    {"name": "count_k1", "mode": "count_only", "k": 1, "description": "k>=1 points below threshold"},
    {"name": "count_k2", "mode": "count_only", "k": 2, "description": "k>=2 points below threshold"},
    {"name": "count_k3", "mode": "count_only", "k": 3, "description": "Original: k>=3 points below threshold"},
    {"name": "severity_0.3", "mode": "severity", "severity_margin": 0.3, "description": "k>=3 OR point < threshold-0.3"},
    {"name": "severity_0.5", "mode": "severity", "severity_margin": 0.5, "description": "k>=3 OR point < threshold-0.5"},
    {"name": "zscore_2.5", "mode": "zscore", "outlier_z_threshold": 2.5, "description": "k>=3 OR z-score < -2.5"},
    {"name": "zscore_3.0", "mode": "zscore", "outlier_z_threshold": 3.0, "description": "k>=3 OR z-score < -3.0"},
    {"name": "hybrid", "mode": "hybrid", "severity_margin": 0.5, "outlier_z_threshold": 3.0, "description": "k>=3 OR severity OR zscore"},
]


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """Compute confusion matrix and derived metrics."""
    tp = int(np.sum(predictions & ground_truth))
    fp = int(np.sum(predictions & ~ground_truth))
    tn = int(np.sum(~predictions & ~ground_truth))
    fn = int(np.sum(~predictions & ground_truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_decision_mode(
    scorer: "FCVAEScorer",
    point_scores: np.ndarray,
    point_predictions: np.ndarray,
    ground_truth: np.ndarray,
    mode_config: Dict,
) -> Dict:
    """
    Evaluate a specific decision mode configuration.

    Args:
        scorer: FCVAEScorer instance (will be modified temporarily)
        point_scores: Per-point NLL scores (N, T)
        point_predictions: Per-point binary predictions (N, T)
        ground_truth: Ground truth labels (N,)
        mode_config: Decision mode configuration dict

    Returns:
        Dict with metrics for this mode
    """
    # Temporarily override scorer config
    original_mode = scorer.config.decision_mode
    original_severity = scorer.config.severity_margin
    original_zscore = scorer.config.outlier_z_threshold

    scorer.config.decision_mode = mode_config["mode"]
    if "severity_margin" in mode_config:
        scorer.config.severity_margin = mode_config["severity_margin"]
    if "outlier_z_threshold" in mode_config:
        scorer.config.outlier_z_threshold = mode_config["outlier_z_threshold"]

    # Get predictions with this mode
    window_predictions = scorer.predict_windows_from_points(point_predictions, point_scores)

    # Compute metrics
    metrics = compute_metrics(window_predictions, ground_truth)
    metrics["name"] = mode_config["name"]
    metrics["description"] = mode_config["description"]
    metrics["num_predicted"] = int(window_predictions.sum())

    # Restore original config
    scorer.config.decision_mode = original_mode
    scorer.config.severity_margin = original_severity
    scorer.config.outlier_z_threshold = original_zscore

    return metrics


def aggregate_to_daily(
    point_scores: np.ndarray,
    timestamps: np.ndarray,
    point_threshold: float,
    mode_config: Dict,
    hours_per_day: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate sliding window scores to daily predictions using specified decision mode.

    Args:
        point_scores: Per-point NLL scores (num_windows, window_size)
        timestamps: Starting hour for each window (must match point_scores length)
        point_threshold: Point-level threshold
        mode_config: Decision mode configuration
        hours_per_day: Hours per day

    Returns:
        Tuple of (daily_predictions, daily_anomaly_counts)
    """
    if len(point_scores) == 0:
        return np.array([]), np.array([])

    # Ensure timestamps matches point_scores
    num_windows = point_scores.shape[0]
    if len(timestamps) != num_windows:
        # Use only the last num_windows timestamps (test windows)
        timestamps = timestamps[-num_windows:]

    window_size = point_scores.shape[1]
    mode = mode_config["mode"]
    severity_margin = mode_config.get("severity_margin", 0.5)
    outlier_z_threshold = mode_config.get("outlier_z_threshold", 3.0)
    k = mode_config.get("k", 3)  # Default to k=3 for backward compatibility

    # Compute per-hour average scores
    min_hour = int(timestamps.min())
    max_hour = int(timestamps.max()) + window_size - 1

    hour_scores = {}
    for window_idx in range(num_windows):
        start_hour = int(timestamps[window_idx])
        for offset in range(window_size):
            hour_idx = start_hour + offset
            if hour_idx not in hour_scores:
                hour_scores[hour_idx] = []
            hour_scores[hour_idx].append(point_scores[window_idx, offset])

    hour_avg_scores = {h: np.mean(scores) for h, scores in hour_scores.items()}

    # Group by calendar day
    first_day = min_hour // hours_per_day
    last_day = max_hour // hours_per_day
    num_days = last_day - first_day + 1

    daily_predictions = np.zeros(num_days, dtype=bool)
    daily_anomaly_counts = np.zeros(num_days, dtype=int)

    for day_offset in range(num_days):
        day_idx = first_day + day_offset
        day_start_hour = day_idx * hours_per_day

        # Collect scores for this day
        day_scores = []
        for hour_of_day in range(hours_per_day):
            hour_idx = day_start_hour + hour_of_day
            if hour_idx in hour_avg_scores:
                day_scores.append(hour_avg_scores[hour_idx])

        if len(day_scores) == 0:
            continue

        day_scores = np.array(day_scores)

        # Count points below threshold
        n_anomalous = np.sum(day_scores < point_threshold)
        daily_anomaly_counts[day_offset] = n_anomalous

        # Apply decision logic
        if mode == "count_only":
            daily_predictions[day_offset] = n_anomalous >= k
        elif mode == "severity":
            count_crit = n_anomalous >= k
            severe_thresh = point_threshold - severity_margin
            severity_crit = np.any(day_scores < severe_thresh)
            daily_predictions[day_offset] = count_crit | severity_crit
        elif mode == "zscore":
            count_crit = n_anomalous >= k
            if len(day_scores) > 1:
                z_scores = (day_scores - np.mean(day_scores)) / (np.std(day_scores) + 1e-8)
                zscore_crit = np.any(z_scores < -outlier_z_threshold)
            else:
                zscore_crit = False
            daily_predictions[day_offset] = count_crit | zscore_crit
        elif mode == "hybrid":
            count_crit = n_anomalous >= k
            severe_thresh = point_threshold - severity_margin
            severity_crit = np.any(day_scores < severe_thresh)
            if len(day_scores) > 1:
                z_scores = (day_scores - np.mean(day_scores)) / (np.std(day_scores) + 1e-8)
                zscore_crit = np.any(z_scores < -outlier_z_threshold)
            else:
                zscore_crit = False
            daily_predictions[day_offset] = count_crit | severity_crit | zscore_crit
        else:
            daily_predictions[day_offset] = n_anomalous >= k

    return daily_predictions, daily_anomaly_counts


def get_daily_ground_truth(
    preprocessor: "TransactionPreprocessor",
    combo: Tuple[str, str],
    test_start_day: int,
    num_test_days: int,
) -> np.ndarray:
    """Extract ground truth daily anomaly labels."""
    hourly_df = preprocessor.combo_hourly.get(combo)
    if hourly_df is None:
        return np.zeros(num_test_days, dtype=bool)

    if "is_anomaly" not in hourly_df.columns:
        return np.zeros(num_test_days, dtype=bool)

    daily_ground_truth = np.zeros(num_test_days, dtype=bool)

    for day_offset in range(num_test_days):
        day_idx = test_start_day + day_offset
        day_start_hour = day_idx * 24
        day_end_hour = day_start_hour + 24

        day_data = hourly_df.iloc[day_start_hour:day_end_hour]
        if len(day_data) > 0 and "is_anomaly" in day_data.columns:
            daily_ground_truth[day_offset] = day_data["is_anomaly"].sum() > 0

    return daily_ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Compare decision logic variants for FCVAE anomaly detection"
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
        default="results/fcvae_decision_logic_comparison",
        help="Directory to save comparison results"
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

    print("\n" + "=" * 80)
    print("FCVAE DECISION LOGIC COMPARISON")
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

    print(f"\nDecision modes to compare:")
    for mode in DECISION_MODES:
        print(f"  - {mode['name']}: {mode['description']}")

    # Evaluate each combo
    all_results = {}

    for combo in COMBO_KEYS:
        if combo not in registry.models:
            logger.warning(f"No model found for {combo}, skipping")
            continue

        combo_name = f"{combo[0]}/{combo[1]}"
        print(f"\n{'='*60}")
        print(f"Evaluating {combo_name}")
        print("=" * 60)

        model = registry.get_model(combo)
        scorer = registry.get_scorer(combo)

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

        # Get timestamps for daily aggregation
        sw_data = preprocessor.combo_sliding_windows.get(combo, {})
        all_timestamps = sw_data.get("timestamps", np.array([]))

        hours_per_day = 24
        train_days = preprocessor.config.train_days
        val_days = preprocessor.config.val_days
        threshold_days = preprocessor.config.threshold_days
        test_start_hour = (train_days + val_days + threshold_days) * hours_per_day
        test_start_day = train_days + val_days + threshold_days

        test_timestamps = all_timestamps[all_timestamps >= test_start_hour]

        # Run one iteration first to get actual num_test_days from daily aggregation
        first_mode = DECISION_MODES[0]
        daily_predictions_first, _ = aggregate_to_daily(
            point_scores=point_scores,
            timestamps=test_timestamps,
            point_threshold=scorer.point_threshold,
            mode_config=first_mode,
            hours_per_day=hours_per_day,
        )
        num_test_days = len(daily_predictions_first)

        # Get the actual test_start_day from timestamps
        if len(test_timestamps) > 0:
            # Use the first timestamp to determine test_start_day
            min_test_hour = int(test_timestamps[-num_test_days * hours_per_day:].min()) if len(test_timestamps) >= num_test_days else int(test_timestamps.min())
            test_start_day = min_test_hour // hours_per_day
        else:
            test_start_day = train_days + val_days + threshold_days

        daily_ground_truth = get_daily_ground_truth(
            preprocessor=preprocessor,
            combo=combo,
            test_start_day=test_start_day,
            num_test_days=num_test_days,
        )

        # Window-level ground truth: a window is anomalous if any point has anomaly label
        window_ground_truth = (test_labels.sum(axis=1) > 0)
        num_anomaly_windows = int(window_ground_truth.sum())

        print(f"  Test windows: {len(test_windows)}")
        print(f"  Anomaly windows (ground truth): {num_anomaly_windows}")
        print(f"  Test days: {num_test_days}")
        print(f"  Ground truth anomaly days: {daily_ground_truth.sum()}")
        print(f"  Point threshold: {scorer.point_threshold:.4f}")

        # Evaluate each decision mode at both window and daily level
        combo_results = []

        for mode_config in DECISION_MODES:
            # WINDOW-LEVEL: Direct prediction on each sliding window
            scorer.config.decision_mode = mode_config["mode"]
            if "k" in mode_config:
                scorer.config.hard_criterion_k = mode_config["k"]
            if "severity_margin" in mode_config:
                scorer.config.severity_margin = mode_config["severity_margin"]
            if "outlier_z_threshold" in mode_config:
                scorer.config.outlier_z_threshold = mode_config["outlier_z_threshold"]

            window_predictions = scorer.predict_windows_from_points(point_predictions, point_scores)
            window_metrics = compute_metrics(window_predictions, window_ground_truth)

            # Reset scorer config for next iteration
            scorer.config.decision_mode = "count_only"
            scorer.config.hard_criterion_k = 3
            scorer.config.severity_margin = 0.5
            scorer.config.outlier_z_threshold = 3.0

            # DAILY-LEVEL: Aggregate to daily
            daily_predictions, daily_anomaly_counts = aggregate_to_daily(
                point_scores=point_scores,
                timestamps=test_timestamps,
                point_threshold=scorer.point_threshold,
                mode_config=mode_config,
                hours_per_day=hours_per_day,
            )

            daily_metrics = compute_metrics(daily_predictions, daily_ground_truth)

            # Combine metrics
            metrics = {
                "name": mode_config["name"],
                "description": mode_config["description"],
                # Window-level metrics
                "w_TP": window_metrics["TP"],
                "w_FP": window_metrics["FP"],
                "w_FN": window_metrics["FN"],
                "w_TN": window_metrics["TN"],
                "w_precision": window_metrics["precision"],
                "w_recall": window_metrics["recall"],
                "w_f1": window_metrics["f1"],
                # Daily-level metrics
                "d_TP": daily_metrics["TP"],
                "d_FP": daily_metrics["FP"],
                "d_FN": daily_metrics["FN"],
                "d_TN": daily_metrics["TN"],
                "d_precision": daily_metrics["precision"],
                "d_recall": daily_metrics["recall"],
                "d_f1": daily_metrics["f1"],
            }
            metrics["name"] = mode_config["name"]
            metrics["description"] = mode_config["description"]
            metrics["num_predicted"] = int(daily_predictions.sum())

            combo_results.append(metrics)

        all_results[combo] = {
            "combo_name": combo_name,
            "num_test_windows": len(test_windows),
            "num_anomaly_windows": num_anomaly_windows,
            "num_test_days": num_test_days,
            "num_ground_truth_days": int(daily_ground_truth.sum()),
            "point_threshold": float(scorer.point_threshold),
            "mode_results": combo_results,
        }

        # Print per-combo results (WINDOW-LEVEL)
        print(f"\n  WINDOW-LEVEL ({len(test_windows)} windows, {num_anomaly_windows} anomaly windows):")
        print(f"  {'Mode':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print(f"  {'-'*76}")
        for r in combo_results:
            print(f"  {r['name']:<20} {r['w_TP']:>4} {r['w_FP']:>4} {r['w_FN']:>4} {r['w_TN']:>4} "
                  f"{r['w_precision']:>8.4f} {r['w_recall']:>8.4f} {r['w_f1']:>8.4f}")

    # Print aggregate summary
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY (across all combos)")
    print("=" * 80)

    # Aggregate metrics across combos - WINDOW LEVEL
    aggregate_window = {}
    for mode_config in DECISION_MODES:
        mode_name = mode_config["name"]
        aggregate_window[mode_name] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        for combo, results in all_results.items():
            for r in results["mode_results"]:
                if r["name"] == mode_name:
                    aggregate_window[mode_name]["TP"] += r["w_TP"]
                    aggregate_window[mode_name]["FP"] += r["w_FP"]
                    aggregate_window[mode_name]["TN"] += r["w_TN"]
                    aggregate_window[mode_name]["FN"] += r["w_FN"]

        # Compute derived metrics
        tp = aggregate_window[mode_name]["TP"]
        fp = aggregate_window[mode_name]["FP"]
        fn = aggregate_window[mode_name]["FN"]
        tn = aggregate_window[mode_name]["TN"]

        aggregate_window[mode_name]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        aggregate_window[mode_name]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p, r = aggregate_window[mode_name]["precision"], aggregate_window[mode_name]["recall"]
        aggregate_window[mode_name]["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print(f"\nWINDOW-LEVEL (comparable to original metrics):")
    print(f"{'Mode':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 82)

    for mode_config in DECISION_MODES:
        mode_name = mode_config["name"]
        m = aggregate_window[mode_name]
        print(f"{mode_name:<20} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5} "
              f"{m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # Find best mode by F1
    best_mode = max(aggregate_window.keys(), key=lambda x: aggregate_window[x]["f1"])
    best_f1 = aggregate_window[best_mode]["f1"]
    print(f"\nBest mode by F1: {best_mode} (F1={best_f1:.4f})")

    # Find best mode by recall (with precision >= 0.8)
    high_precision_modes = {k: v for k, v in aggregate_window.items() if v["precision"] >= 0.8}
    if high_precision_modes:
        best_recall_mode = max(high_precision_modes.keys(), key=lambda x: high_precision_modes[x]["recall"])
        print(f"Best recall with P>=0.8: {best_recall_mode} (R={aggregate_window[best_recall_mode]['recall']:.4f}, F1={aggregate_window[best_recall_mode]['f1']:.4f})")

    # Save results to file
    results_path = output_path / "decision_logic_comparison.txt"
    with open(results_path, "w") as f:
        f.write("FCVAE Decision Logic Comparison Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("WINDOW-LEVEL AGGREGATE SUMMARY (comparable to original metrics)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Mode':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("-" * 82 + "\n")

        for mode_config in DECISION_MODES:
            mode_name = mode_config["name"]
            m = aggregate_window[mode_name]
            f.write(f"{mode_name:<20} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5} "
                    f"{m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}\n")

        f.write(f"\nBest mode by F1: {best_mode} (F1={best_f1:.4f})\n")

        f.write("\n\nPER-COMBO BREAKDOWN (WINDOW-LEVEL)\n")
        f.write("=" * 80 + "\n")

        for combo, results in all_results.items():
            f.write(f"\n{results['combo_name']}\n")
            f.write(f"  Windows: {results['num_test_windows']}, Anomaly windows: {results['num_anomaly_windows']}\n")
            f.write(f"  {'Mode':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Prec':>8} {'Recall':>8} {'F1':>8}\n")
            f.write(f"  {'-'*76}\n")
            for r in results["mode_results"]:
                f.write(f"  {r['name']:<20} {r['w_TP']:>4} {r['w_FP']:>4} {r['w_FN']:>4} {r['w_TN']:>4} "
                        f"{r['w_precision']:>8.4f} {r['w_recall']:>8.4f} {r['w_f1']:>8.4f}\n")

    print(f"\nResults saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
