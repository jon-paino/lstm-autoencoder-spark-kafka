"""
Step 1: Train Model -- baseline run

Trains an LSTM Encoder-Decoder on the NYC taxi CSV and saves artifacts
to `models/initial/`. Defaults are intentionally a small, fast
configuration (hidden_dim=24, epochs=20, lr=2e-3, train_weeks=6) so the
baseline catches all 5 known anomalies but trips on 2 normal weeks
(F1 ~ 83%). Step 4 (`code/4_grid_sweep.py`) then improves on it by
searching for a better hyperparameter / split combination and
retraining.

Window-level Mahalanobis scoring on the full 336-step error vector;
threshold at the 99.99th percentile of validation distances. See
`notebooks/model_design.ipynb` for the motivation.

The prebuilt reference artifacts at `models/lstm_model.pt`,
`models/scaler.pkl`, `models/scorer.pkl` are NEVER overwritten.

Usage:
    python code/1_train_model.py
    python code/1_train_model.py --epochs 100 --hidden-dim 64 --lr 5e-4
    python code/1_train_model.py --output-dir models/my_run
"""

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import EncDecAD, ModelConfig, create_model
from src.scorer import AnomalyScorer, ScorerConfig
from src.preprocess import (
    PreprocessorConfig,
    TimeSeriesDataset,
    preprocess_pipeline,
    get_test_week_info,
)
from src.synthetic import SyntheticAnomalyConfig, generate_synthetic_dataset
from src.training import TrainingConfig, save_training_artifacts, train_model

# Register old module paths so pickled artifacts (saved with old names) can be loaded
import src.model, src.scorer
sys.modules["lstm_autoencoder"] = src.model
sys.modules["anomaly_scorer"] = src.scorer

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> EncDecAD:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EncDecAD(config=checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Encoder-Decoder for Anomaly Detection")
    parser.add_argument("--data-path", type=str, default=str(PROJECT_ROOT / "data" / "nyc_taxi.csv"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "models" / "initial"),
                        help="Where to save your trained artifacts. Defaults to models/initial/ to leave the prebuilt models/lstm_model.pt et al untouched.")
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold-percentile", type=float, default=99.99)
    parser.add_argument("--train-weeks", type=int, default=6)
    parser.add_argument("--val-weeks", type=int, default=2)
    parser.add_argument("--threshold-weeks", type=int, default=2)
    parser.add_argument("--use-synthetic-anomalies", action="store_true",
                        help="Calibrate the threshold against synthetic anomalies instead of the 99.99th percentile of normal validation scores.")
    parser.add_argument("--synthetic-anomaly-types", type=str, nargs="+", default=["point", "level_shift"])
    parser.add_argument("--threshold-calibration-method", type=str, default="midpoint",
                        choices=["midpoint", "f1_max", "youden", "percentile"])
    parser.add_argument("--synthetic-magnitude", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Determinism: every run with the same --seed should produce the same artifacts.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "=" * 60)
    print("LSTM ENCODER-DECODER TRAINING -- baseline run")
    print("=" * 60)
    print(f"Output directory:   {args.output_dir}")
    print(f"Hidden dim:         {args.hidden_dim}")
    print(f"Learning rate:      {args.lr}")
    print(f"Max epochs:         {args.epochs}")
    print(f"Scoring:            window-level Mahalanobis @ {args.threshold_percentile}th percentile")
    print()
    print("This is a fast baseline -- expect F1 around 83% (catches all 5 known")
    print("anomalies but trips on 2 normal weeks). Run")
    print("    python code/4_grid_sweep.py")
    print("next to search for a better configuration. The sweep will retrain the")
    print("winning config and save it to models/best/.")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Step 1: Preprocess data
    print("\n--- Step 1: Preprocessing data ---")
    preprocess_config = PreprocessorConfig(
        train_weeks=args.train_weeks,
        val_weeks=args.val_weeks,
        threshold_weeks=args.threshold_weeks,
    )
    dataloaders, normalized_splits, scaler, week_info, split_indices = preprocess_pipeline(
        args.data_path, config=preprocess_config, batch_size=args.batch_size
    )

    # Step 2: Create model
    print("\n--- Step 2: Creating model ---")
    model = create_model(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.to(device)
    print(f"Model config: {model.get_config()}")

    # Step 3: Train model
    print("\n--- Step 3: Training model ---")
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
    )
    model, history = train_model(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=device,
        config=training_config,
    )

    # Step 4: Fit anomaly scorer (window-level Mahalanobis on validation errors)
    print("\n--- Step 4: Fitting anomaly scorer (window-level Mahalanobis) ---")
    scorer_config = ScorerConfig(
        threshold_percentile=args.threshold_percentile,
        scoring_mode="window",
    )
    scorer = AnomalyScorer(config=scorer_config)
    scorer.fit(model, dataloaders["val"], device)

    if args.use_synthetic_anomalies:
        print("\nUsing synthetic anomalies for threshold calibration")
        synth_config = SyntheticAnomalyConfig(
            anomaly_types=args.synthetic_anomaly_types,
            num_synthetic_per_normal=1,
            point_magnitude=args.synthetic_magnitude,
            level_shift_magnitude=args.synthetic_magnitude,
            noise_scale=args.synthetic_magnitude * 0.6,
        )
        synthetic_weeks, _, _ = generate_synthetic_dataset(
            normalized_splits["threshold_val"], config=synth_config
        )
        synthetic_dataset = TimeSeriesDataset(synthetic_weeks)
        synthetic_loader = DataLoader(synthetic_dataset, batch_size=args.batch_size, shuffle=False)

        normal_scores, _ = scorer.compute_scores(model, dataloaders["threshold_val"], device)
        synth_scores, _ = scorer.compute_scores(model, synthetic_loader, device)
        if args.threshold_calibration_method != "percentile":
            opt_t, _ = scorer.find_optimal_threshold(
                normal_scores, synth_scores, method=args.threshold_calibration_method
            )
            scorer.threshold = opt_t
        else:
            scorer.set_threshold(normal_scores)
    else:
        val_scores, _ = scorer.compute_scores(model, dataloaders["threshold_val"], device)
        scorer.set_threshold(val_scores)

    # Step 5: Quick evaluation on test set
    print("\n--- Step 5: Evaluating on test set ---")
    test_info = get_test_week_info(week_info, split_indices)

    test_scores, _ = scorer.compute_scores(model, dataloaders["test"], device)
    predictions = scorer.predict(test_scores)

    print(f"\n{'Week':<10} {'Score':>12} {'Predicted':>10} {'Actual':>12} {'Match':>6}")
    print("-" * 55)

    for score, pred, week in zip(test_scores, predictions, test_info):
        if week.get("is_edge_case"):
            continue
        pred_str = "ANOMALY" if pred else "normal"
        actual_str = "ANOMALY" if week["is_anomaly"] else "normal"
        match = "Y" if pred == week["is_anomaly"] else "N"
        print(f"{week['year_week']:<12} {score:>12.2f} {pred_str:>10} {actual_str:>12} {match:>6}")

    scored = [(p, w) for p, w in zip(predictions, test_info) if not w.get("is_edge_case")]
    tp = sum(1 for p, w in scored if p and w["is_anomaly"])
    fp = sum(1 for p, w in scored if p and not w["is_anomaly"])
    fn = sum(1 for p, w in scored if not p and w["is_anomaly"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.2%}  Recall: {recall:.2%}  F1: {f1:.2%}")

    # Step 6: Save artifacts
    print("\n--- Step 6: Saving artifacts ---")
    save_training_artifacts(
        output_dir=args.output_dir,
        model=model,
        scaler=scaler,
        scorer=scorer,
        history=history,
        preprocess_config=preprocess_config,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nArtifacts saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
