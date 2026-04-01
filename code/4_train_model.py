"""
Step 4: Train Model
Full training pipeline: preprocess -> train -> fit scorer -> save artifacts.

Usage:
    python code/4_train_model.py
    python code/4_train_model.py --epochs 50 --lr 0.001
    python code/4_train_model.py --use-synthetic-anomalies
"""

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
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
    get_test_timestamps,
)
from src.synthetic import SyntheticAnomalyConfig, generate_synthetic_dataset

# Register old module paths so pickled artifacts (saved with old names) can be loaded
import src.model, src.scorer
sys.modules["lstm_autoencoder"] = src.model
sys.modules["anomaly_scorer"] = src.scorer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    learning_rate: float = 1e-3
    patience: int = 10
    min_delta: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_model(
    model: EncDecAD,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Optional[TrainingConfig] = None,
) -> Tuple[EncDecAD, Dict]:
    """
    Train the LSTM Encoder-Decoder model.

    Returns:
        Tuple of (trained_model, training_history)
    """
    config = config or TrainingConfig()

    logger.info("Starting training...")
    logger.info(f"  Epochs: {config.epochs}, LR: {config.learning_rate}, Patience: {config.patience}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    best_model_state = None
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            x_reconstructed = model(x)
            loss = criterion(x_reconstructed, x)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        train_loss /= num_batches

        # Validate
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                x_reconstructed = model(x)
                loss = criterion(x_reconstructed, x)
                val_loss += loss.item()
                num_val_batches += 1
        val_loss /= num_val_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch + 1

        improved = "<-" if val_loss <= best_val_loss else ""
        logger.info(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} {improved}"
        )

        if early_stopping.step(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        logger.info(f"Restored best model from epoch {history['best_epoch']}")

    return model, history


def save_training_artifacts(
    output_dir: str,
    model: EncDecAD,
    scaler,
    scorer: AnomalyScorer,
    history: Dict,
    preprocess_config: PreprocessorConfig,
) -> None:
    """Save all training artifacts for deployment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = output_dir / "lstm_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
    }, model_path)
    logger.info(f"Saved model to {model_path}")

    # Scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")

    # Scorer
    scorer.save(str(output_dir / "scorer.pkl"))

    # Training history
    with open(output_dir / "training_history.pkl", "wb") as f:
        pickle.dump(history, f)
    logger.info(f"Saved training history")

    # Preprocessor config
    with open(output_dir / "preprocessor_config.pkl", "wb") as f:
        pickle.dump(preprocess_config, f)
    logger.info(f"Saved preprocessor config")


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
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold-percentile", type=float, default=99.99)
    parser.add_argument("--train-weeks", type=int, default=9)
    parser.add_argument("--val-weeks", type=int, default=3)
    parser.add_argument("--threshold-weeks", type=int, default=2)
    parser.add_argument("--scoring-mode", type=str, default="point", choices=["point", "window"])
    parser.add_argument("--hard-criterion-k", type=int, default=5)
    parser.add_argument("--use-synthetic-anomalies", action="store_true")
    parser.add_argument("--synthetic-anomaly-types", type=str, nargs="+", default=["point", "level_shift"])
    parser.add_argument("--threshold-calibration-method", type=str, default="midpoint",
                        choices=["midpoint", "f1_max", "youden", "percentile"])
    parser.add_argument("--synthetic-magnitude", type=float, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n" + "=" * 60)
    print("LSTM ENCODER-DECODER TRAINING")
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

    # Step 4: Fit anomaly scorer
    print(f"\n--- Step 4: Fitting anomaly scorer (mode={args.scoring_mode}) ---")
    scorer_config = ScorerConfig(
        threshold_percentile=args.threshold_percentile,
        scoring_mode=args.scoring_mode,
        hard_criterion_k=args.hard_criterion_k,
    )
    scorer = AnomalyScorer(config=scorer_config)
    scorer.fit(model, dataloaders["val"], device)

    if args.use_synthetic_anomalies:
        print(f"\nUsing synthetic anomalies for threshold calibration")
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

        if args.scoring_mode == "point":
            normal_ps, normal_ws, _ = scorer.compute_point_scores(model, dataloaders["threshold_val"], device)
            synth_ps, synth_ws, _ = scorer.compute_point_scores(model, synthetic_loader, device)

            if args.threshold_calibration_method != "percentile":
                opt_pt, _ = scorer.find_optimal_threshold(normal_ps.flatten(), synth_ps.flatten(), method=args.threshold_calibration_method)
                scorer.point_threshold = opt_pt
                opt_wt, _ = scorer.find_optimal_threshold(normal_ws, synth_ws, method=args.threshold_calibration_method)
                scorer.threshold = opt_wt
            else:
                scorer.set_point_threshold(normal_ps)
                scorer.set_threshold(normal_ws)
        else:
            normal_scores, _ = scorer.compute_scores(model, dataloaders["threshold_val"], device)
            synth_scores, _ = scorer.compute_scores(model, synthetic_loader, device)
            if args.threshold_calibration_method != "percentile":
                opt_t, _ = scorer.find_optimal_threshold(normal_scores, synth_scores, method=args.threshold_calibration_method)
                scorer.threshold = opt_t
            else:
                scorer.set_threshold(normal_scores)
    else:
        if args.scoring_mode == "point":
            point_scores, window_scores, _ = scorer.compute_point_scores(model, dataloaders["threshold_val"], device)
            scorer.set_point_threshold(point_scores)
            scorer.set_threshold(window_scores)
        else:
            val_scores, _ = scorer.compute_scores(model, dataloaders["threshold_val"], device)
            scorer.set_threshold(val_scores)

    # Step 5: Quick evaluation on test set
    print("\n--- Step 5: Evaluating on test set ---")
    test_info = get_test_week_info(week_info, split_indices)
    test_timestamps = get_test_timestamps(week_info, split_indices)

    if args.scoring_mode == "point":
        point_scores, window_scores, _ = scorer.compute_point_scores(model, dataloaders["test"], device)
        point_preds = scorer.predict_points(point_scores)
        predictions = scorer.predict_windows_from_points(point_preds)
        test_scores = window_scores
    else:
        test_scores, _ = scorer.compute_scores(model, dataloaders["test"], device)
        predictions = scorer.predict(test_scores)

    print(f"\n{'Week':<10} {'Score':>12} {'Predicted':>10} {'Actual':>12} {'Match':>6}")
    print("-" * 55)

    correct = 0
    for i, (score, pred, week) in enumerate(zip(test_scores, predictions, test_info)):
        pred_str = "ANOMALY" if pred else "normal"
        actual_str = "ANOMALY" if week["is_anomaly"] else "normal"
        match = pred == week["is_anomaly"]
        if match:
            correct += 1
        print(f"{week['year_week']:<10} {score:>12.2f} {pred_str:>10} {actual_str:>12} {'Y' if match else 'N':>6}")

    tp = sum(1 for p, w in zip(predictions, test_info) if p and w["is_anomaly"])
    fp = sum(1 for p, w in zip(predictions, test_info) if p and not w["is_anomaly"])
    fn = sum(1 for p, w in zip(predictions, test_info) if not p and w["is_anomaly"])
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
