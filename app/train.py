"""
Training Pipeline for LSTM Encoder-Decoder Anomaly Detection

Implements the training loop with:
- MSE loss for reconstruction
- Adam optimizer
- Early stopping based on validation loss
- Model checkpointing
- Training history logging
"""

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocessor import NYCTaxiPreprocessor, PreprocessorConfig
from lstm_autoencoder import EncDecAD, ModelConfig, create_model
from anomaly_scorer import AnomalyScorer, ScorerConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    learning_rate: float = 1e-3
    patience: int = 10          # Early stopping patience
    min_delta: float = 1e-6     # Minimum improvement for early stopping
    weight_decay: float = 0.0   # L2 regularization
    grad_clip: float = 1.0      # Gradient clipping max norm


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
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
    config: Optional[TrainingConfig] = None
) -> Tuple[EncDecAD, Dict]:
    """
    Train the LSTM Encoder-Decoder model.

    Args:
        model: EncDecAD model to train
        train_loader: DataLoader with training sequences
        val_loader: DataLoader with validation sequences
        device: Device to train on (cpu/cuda)
        config: TrainingConfig with hyperparameters

    Returns:
        Tuple of (trained_model, training_history)
    """
    config = config or TrainingConfig()

    logger.info("Starting training...")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Patience: {config.patience}")
    logger.info(f"  Device: {device}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta
    )

    # Track best model
    best_model_state = None
    best_val_loss = float("inf")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
    }

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            x_reconstructed = model(x)
            loss = criterion(x_reconstructed, x)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip
                )

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Validation phase
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

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch + 1

        # Logging
        improved = "←" if val_loss <= best_val_loss else ""
        logger.info(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} {improved}"
        )

        # Early stopping check
        if early_stopping.step(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        logger.info(f"Restored best model from epoch {history['best_epoch']}")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

    return model, history


def save_training_artifacts(
    output_dir: str,
    model: EncDecAD,
    preprocessor: NYCTaxiPreprocessor,
    scorer: AnomalyScorer,
    history: Dict
) -> None:
    """
    Save all training artifacts for deployment.

    Args:
        output_dir: Directory to save artifacts
        model: Trained model
        preprocessor: Fitted preprocessor (with scaler)
        scorer: Fitted anomaly scorer
        history: Training history
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "lstm_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
    }, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(preprocessor.scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")

    # Save scorer
    scorer_path = output_dir / "scorer.pkl"
    scorer.save(scorer_path)

    # Save training history
    history_path = output_dir / "training_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    logger.info(f"Saved training history to {history_path}")

    # Save preprocessor config (data split configuration)
    config_path = output_dir / "preprocessor_config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(preprocessor.config, f)
    logger.info(f"Saved preprocessor config to {config_path}")


def load_model(model_path: str, device: torch.device) -> EncDecAD:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model to

    Returns:
        Loaded EncDecAD model
    """
    # Allow our custom ModelConfig class for safe loading
    torch.serialization.add_safe_globals([ModelConfig])

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model = EncDecAD(config=checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    return model


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train LSTM Encoder-Decoder for Anomaly Detection"
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
        default="models",
        help="Directory to save trained model and artifacts"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="LSTM hidden dimension"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of LSTM layers"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.99,
        help="Percentile for anomaly threshold"
    )
    parser.add_argument(
        "--train-weeks",
        type=int,
        default=8,
        help="Number of normal weeks for training"
    )
    parser.add_argument(
        "--val-weeks",
        type=int,
        default=2,
        help="Number of normal weeks for early stopping validation"
    )
    parser.add_argument(
        "--threshold-weeks",
        type=int,
        default=4,
        help="Number of normal weeks for threshold calibration"
    )
    parser.add_argument(
        "--scoring-mode",
        type=str,
        default="point",
        choices=["point", "window"],
        help="Scoring mode: 'point' (Malhotra paper) or 'window' (legacy)"
    )
    parser.add_argument(
        "--hard-criterion-k",
        type=int,
        default=5,
        help="Number of anomalous points to flag window as anomalous (HardCriterion)"
    )
    # Synthetic anomaly calibration arguments
    parser.add_argument(
        "--use-synthetic-anomalies",
        action="store_true",
        help="Use synthetic anomalies for threshold calibration (improves threshold selection)"
    )
    parser.add_argument(
        "--synthetic-anomaly-types",
        type=str,
        nargs="+",
        default=["point", "level_shift"],
        help="Types of synthetic anomalies to generate (point, level_shift, noise, temporal)"
    )
    parser.add_argument(
        "--threshold-calibration-method",
        type=str,
        default="midpoint",
        choices=["midpoint", "f1_max", "youden", "percentile"],
        help="Method for threshold calibration when using synthetic anomalies"
    )
    parser.add_argument(
        "--synthetic-magnitude",
        type=float,
        default=1,
        help="Base magnitude for synthetic anomalies (in std units)"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("LSTM ENCODER-DECODER TRAINING")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Step 1: Preprocess data
    print("\n" + "-" * 40)
    print("Step 1: Preprocessing data")
    print("-" * 40)

    config = PreprocessorConfig(
        train_weeks=args.train_weeks,
        val_weeks=args.val_weeks,
        threshold_weeks=args.threshold_weeks
    )
    preprocessor = NYCTaxiPreprocessor(config=config)
    dataloaders, normalized_splits = preprocessor.preprocess(
        args.data_path,
        batch_size=args.batch_size
    )

    # Step 2: Create model
    print("\n" + "-" * 40)
    print("Step 2: Creating model")
    print("-" * 40)

    model = create_model(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model.to(device)

    print(f"Model config: {model.get_config()}")

    # Step 3: Train model
    print("\n" + "-" * 40)
    print("Step 3: Training model")
    print("-" * 40)

    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience
    )

    model, history = train_model(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=device,
        config=training_config
    )

    # Step 4: Fit anomaly scorer
    print("\n" + "-" * 40)
    print(f"Step 4: Fitting anomaly scorer (mode={args.scoring_mode})")
    print("-" * 40)

    scorer_config = ScorerConfig(
        threshold_percentile=args.threshold_percentile,
        scoring_mode=args.scoring_mode,
        hard_criterion_k=args.hard_criterion_k,
    )
    scorer = AnomalyScorer(config=scorer_config)

    # Fit error distribution on validation data (paper recommendation: vN1)
    # The model hasn't optimized on val data, so errors there are more realistic
    scorer.fit(model, dataloaders["val"], device)

    if args.use_synthetic_anomalies:
        # Synthetic anomaly calibration: generate synthetic anomalies and find optimal threshold
        from synthetic_anomaly import SyntheticAnomalyGenerator, SyntheticAnomalyConfig
        from torch.utils.data import DataLoader as TorchDataLoader

        print(f"\nUsing synthetic anomalies for threshold calibration")
        print(f"  Anomaly types: {args.synthetic_anomaly_types}")
        print(f"  Calibration method: {args.threshold_calibration_method}")

        # Configure synthetic anomaly generation
        synth_config = SyntheticAnomalyConfig(
            anomaly_types=args.synthetic_anomaly_types,
            num_synthetic_per_normal=1,
            point_magnitude=args.synthetic_magnitude,
            level_shift_magnitude=args.synthetic_magnitude,
            noise_scale=args.synthetic_magnitude * 0.6,
        )
        generator = SyntheticAnomalyGenerator(synth_config)

        # Generate synthetic anomalies from threshold_val normal weeks
        synthetic_weeks, _, anomaly_types = generator.generate_synthetic_dataset(
            normalized_splits["threshold_val"]
        )

        # Create synthetic DataLoader
        from data_preprocessor import TimeSeriesDataset
        synthetic_dataset = TimeSeriesDataset(synthetic_weeks)
        synthetic_loader = TorchDataLoader(
            synthetic_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        if args.scoring_mode == "point":
            # Compute point-level scores for both normal and synthetic
            normal_point_scores, normal_window_scores, _ = scorer.compute_point_scores(
                model, dataloaders["threshold_val"], device
            )
            synth_point_scores, synth_window_scores, _ = scorer.compute_point_scores(
                model, synthetic_loader, device
            )

            if args.threshold_calibration_method != "percentile":
                # Find optimal point-level threshold using labeled data
                optimal_point_threshold, point_metrics = scorer.find_optimal_threshold(
                    normal_point_scores.flatten(),
                    synth_point_scores.flatten(),
                    method=args.threshold_calibration_method
                )
                scorer.point_threshold = optimal_point_threshold
                print(f"  Point threshold (optimal): {optimal_point_threshold:.4f}")
                print(f"  Calibration metrics: {point_metrics}")

                # Find optimal window-level threshold
                optimal_window_threshold, window_metrics = scorer.find_optimal_threshold(
                    normal_window_scores,
                    synth_window_scores,
                    method=args.threshold_calibration_method
                )
                scorer.threshold = optimal_window_threshold
                print(f"  Window threshold (optimal): {optimal_window_threshold:.4f}")
            else:
                # Fall back to percentile-based threshold
                scorer.set_point_threshold(normal_point_scores)
                scorer.set_threshold(normal_window_scores)
        else:
            # Window-level scoring with synthetic calibration
            normal_scores, _ = scorer.compute_scores(
                model, dataloaders["threshold_val"], device
            )
            synth_scores, _ = scorer.compute_scores(
                model, synthetic_loader, device
            )

            if args.threshold_calibration_method != "percentile":
                optimal_threshold, metrics = scorer.find_optimal_threshold(
                    normal_scores,
                    synth_scores,
                    method=args.threshold_calibration_method
                )
                scorer.threshold = optimal_threshold
                print(f"  Window threshold (optimal): {optimal_threshold:.4f}")
                print(f"  Calibration metrics: {metrics}")
            else:
                scorer.set_threshold(normal_scores)

    else:
        # Standard percentile-based threshold calibration (original behavior)
        if args.scoring_mode == "point":
            # Point-level scoring (Malhotra et al. 2016)
            point_scores, window_scores, _ = scorer.compute_point_scores(
                model, dataloaders["threshold_val"], device
            )
            scorer.set_point_threshold(point_scores)
            # Also set window-level threshold for max-score aggregation
            scorer.set_threshold(window_scores)
        else:
            # Legacy window-level scoring
            val_scores, _ = scorer.compute_scores(
                model, dataloaders["threshold_val"], device
            )
            scorer.set_threshold(val_scores)

    # Step 5: Evaluate on test set
    print("\n" + "-" * 40)
    print("Step 5: Evaluating on test set")
    print("-" * 40)

    # Get test week info and timestamps
    test_week_info = preprocessor.get_test_week_info()
    test_timestamps = preprocessor.get_test_timestamps()

    if args.scoring_mode == "point":
        # Point-level scoring
        point_scores, window_scores, _ = scorer.compute_point_scores(
            model, dataloaders["test"], device
        )
        point_predictions = scorer.predict_points(point_scores)
        predictions = scorer.predict_windows_from_points(point_predictions)

        # Report point-level statistics
        total_points = point_predictions.size
        anomalous_points = point_predictions.sum()
        print(f"\nPoint-Level Statistics:")
        print(f"  Total points: {total_points}")
        print(f"  Anomalous points: {anomalous_points} ({100*anomalous_points/total_points:.2f}%)")
        print(f"  Point threshold: {scorer.point_threshold:.4f}")
        print(f"  HardCriterion k: {args.hard_criterion_k}")

        # Use window_scores for display
        test_scores = window_scores
        # Store point_scores for localization
        all_point_scores = point_scores
    else:
        # Legacy window-level scoring
        test_scores, test_errors = scorer.compute_scores(
            model, dataloaders["test"], device
        )
        predictions = scorer.predict(test_scores)
        # Use raw reconstruction errors for localization (not squared normalized)
        all_point_scores = test_errors

    print("\nTest Results (Window-Level with Localization):")
    print(f"{'Week':<10} {'Score':>12} {'Predicted':>10} {'Actual':>12} {'Match':>6}  {'Localization':<35}")
    print("-" * 90)

    correct = 0
    for i, (score, pred, week) in enumerate(zip(test_scores, predictions, test_week_info)):
        pred_str = "ANOMALY" if pred else "normal"
        actual_str = "ANOMALY" if week["is_anomaly"] else "normal"
        match = pred == week["is_anomaly"]
        match_str = "✓" if match else "✗"
        if match:
            correct += 1

        # Compute localization for anomalous weeks
        loc_str = ""
        if pred:
            timestamps = test_timestamps[i]
            localization = scorer.localize_anomaly(all_point_scores[i], timestamps)
            # Format: "Nov 27 02:30-08:30 (6h, ρ=9.9)"
            start_ts = localization["anomaly_start"]
            end_ts = localization["anomaly_end"]
            scale_h = localization["scale_hours"]
            rho = localization["contrast_ratio"]
            # Extract just date and time for display
            start_short = start_ts[5:16].replace("T", " ") if "T" in start_ts else start_ts[5:16]
            end_short = end_ts[11:16] if "T" in end_ts else end_ts[11:16]
            loc_str = f"{start_short}-{end_short} ({scale_h}h, ρ={rho:.1f})"

        print(f"{week['year_week']:<10} {score:>12.2f} {pred_str:>10} {actual_str:>12} {match_str:>6}  {loc_str:<35}")

    accuracy = correct / len(predictions)
    print("-" * 52)
    print(f"Accuracy: {correct}/{len(predictions)} ({accuracy:.1%})")

    # Print raw error statistics for anomalous weeks (for threshold calibration)
    print("\n" + "=" * 70)
    print("Raw Reconstruction Error Statistics for ANOMALOUS weeks:")
    print("=" * 70)
    for i, week in enumerate(test_week_info):
        if week["is_anomaly"]:
            errors = all_point_scores[i]
            peak_idx = np.argmax(errors)
            print(f"{week['year_week']}: min={errors.min():.4f}, max={errors.max():.4f}, "
                  f"mean={errors.mean():.4f}, std={errors.std():.4f}")
            print(f"  Points >0.2: {np.sum(errors > 0.2):3d}, >0.5: {np.sum(errors > 0.5):3d}, "
                  f">1.0: {np.sum(errors > 1.0):3d}, >2.0: {np.sum(errors > 2.0):3d}")
            print(f"  Peak at index {peak_idx} (hour {peak_idx/2:.1f}), value={errors[peak_idx]:.4f}")
            print()

    # Calculate precision/recall for anomaly class
    true_positives = sum(1 for p, w in zip(predictions, test_week_info)
                        if p and w["is_anomaly"])
    false_positives = sum(1 for p, w in zip(predictions, test_week_info)
                         if p and not w["is_anomaly"])
    false_negatives = sum(1 for p, w in zip(predictions, test_week_info)
                         if not p and w["is_anomaly"])

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nWindow-Level Anomaly Detection Metrics:")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1-Score: {f1:.2%}")

    # Step 6: Save artifacts
    print("\n" + "-" * 40)
    print("Step 6: Saving artifacts")
    print("-" * 40)

    save_training_artifacts(
        output_dir=args.output_dir,
        model=model,
        preprocessor=preprocessor,
        scorer=scorer,
        history=history
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nArtifacts saved to: {args.output_dir}/")
    print("  - lstm_model.pt")
    print("  - scaler.pkl")
    print("  - scorer.pkl")
    print("  - training_history.pkl")
    print("  - preprocessor_config.pkl")


if __name__ == "__main__":
    main()
