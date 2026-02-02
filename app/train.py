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
        default=95.0,
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
    print("Step 4: Fitting anomaly scorer")
    print("-" * 40)

    scorer = AnomalyScorer(
        config=ScorerConfig(threshold_percentile=args.threshold_percentile)
    )

    # Fit error distribution on validation data (paper recommendation: vN1)
    # The model hasn't optimized on val data, so errors there are more realistic
    scorer.fit(model, dataloaders["val"], device)

    # Set threshold using percentile method on normal validation data
    val_scores, _ = scorer.compute_scores(
        model, dataloaders["threshold_val"], device
    )
    scorer.set_threshold(val_scores)

    # Step 5: Evaluate on test set
    print("\n" + "-" * 40)
    print("Step 5: Evaluating on test set")
    print("-" * 40)

    test_scores, test_errors = scorer.compute_scores(
        model, dataloaders["test"], device
    )
    predictions = scorer.predict(test_scores)

    # Get test week info
    test_week_info = preprocessor.get_test_week_info()

    print("\nTest Results:")
    print(f"{'Week':<10} {'Score':>12} {'Predicted':>10} {'Actual':>12} {'Match':>6}")
    print("-" * 52)

    correct = 0
    for score, pred, week in zip(test_scores, predictions, test_week_info):
        pred_str = "ANOMALY" if pred else "normal"
        actual_str = "ANOMALY" if week["is_anomaly"] else "normal"
        match = pred == week["is_anomaly"]
        match_str = "✓" if match else "✗"
        if match:
            correct += 1

        print(f"{week['year_week']:<10} {score:>12.2f} {pred_str:>10} {actual_str:>12} {match_str:>6}")

    accuracy = correct / len(predictions)
    print("-" * 52)
    print(f"Accuracy: {correct}/{len(predictions)} ({accuracy:.1%})")

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

    print(f"\nAnomaly Detection Metrics:")
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
