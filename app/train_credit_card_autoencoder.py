"""
Training script for credit card fraud detection autoencoder.

Trains a feedforward autoencoder on ALL data (fraud + legitimate) to learn
a general compressed representation of the transaction manifold.

Pattern adapted from train.py but for tabular data with per-class tracking.
"""

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from credit_card_preprocessor import CreditCardPreprocessor, CreditCardPreprocessorConfig
from feedforward_autoencoder import AutoencoderConfig, FeedforwardAutoencoder, SemiSupervisedAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for autoencoder training."""

    epochs: int = 100
    learning_rate: float = 1e-4      # Paper's recommendation
    batch_size: int = 256            # Larger than LSTM (4) due to more data
    patience: int = 10               # Early stopping patience
    min_delta: float = 1e-6          # Minimum improvement threshold
    weight_decay: float = 0.0        # L2 regularization (0 = none)
    grad_clip: float = 1.0           # Gradient clipping max norm


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training if no improvement
    for 'patience' consecutive epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Update early stopping state.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def evaluate_autoencoder(
    model: FeedforwardAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> float:
    """
    Evaluate autoencoder reconstruction loss.

    Args:
        model: Trained autoencoder
        dataloader: Validation or test DataLoader
        device: cpu/cuda
        criterion: Loss function (MSELoss)

    Returns:
        Average reconstruction loss
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)

            # Forward pass
            reconstructed = model(features)
            loss = criterion(reconstructed, features)

            # Accumulate
            total_loss += loss.item() * len(features)
            num_samples += len(features)

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss


def train_autoencoder(
    model: FeedforwardAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig
) -> Tuple[FeedforwardAutoencoder, Dict]:
    """
    Train autoencoder with per-class reconstruction tracking.

    Key difference from LSTM training: Trains on ALL data (fraud + legitimate),
    not just normal data. The goal is to learn a general compressed representation.

    Args:
        model: FeedforwardAutoencoder to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: cpu/cuda
        config: Training hyperparameters

    Returns:
        (trained_model, history)
    """
    logger.info("=" * 80)
    logger.info("Training Feedforward Autoencoder")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Early stopping patience: {config.patience}")
    logger.info("=" * 80)

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

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_loss_fraud': [],       # Track fraud reconstruction separately
        'train_loss_legitimate': [],  # Track legitimate reconstruction separately
        'best_epoch': 0
    }

    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(config.epochs):
        # ===== Training Phase =====
        model.train()
        train_loss_total = 0.0
        train_loss_fraud = 0.0
        train_loss_legit = 0.0
        num_fraud = 0
        num_legit = 0
        num_samples = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(features)
            loss = criterion(reconstructed, features)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

            # Track overall loss
            batch_size = len(features)
            train_loss_total += loss.item() * batch_size
            num_samples += batch_size

            # Track per-class reconstruction error
            fraud_mask = labels == 1
            legit_mask = labels == 0

            if fraud_mask.sum() > 0:
                with torch.no_grad():
                    fraud_loss = criterion(
                        reconstructed[fraud_mask],
                        features[fraud_mask]
                    )
                    train_loss_fraud += fraud_loss.item() * fraud_mask.sum().item()
                    num_fraud += fraud_mask.sum().item()

            if legit_mask.sum() > 0:
                with torch.no_grad():
                    legit_loss = criterion(
                        reconstructed[legit_mask],
                        features[legit_mask]
                    )
                    train_loss_legit += legit_loss.item() * legit_mask.sum().item()
                    num_legit += legit_mask.sum().item()

        # Compute epoch averages
        train_loss = train_loss_total / num_samples
        fraud_loss_avg = train_loss_fraud / num_fraud if num_fraud > 0 else 0
        legit_loss_avg = train_loss_legit / num_legit if num_legit > 0 else 0

        # ===== Validation Phase =====
        val_loss = evaluate_autoencoder(model, val_loader, device, criterion)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_loss_fraud'].append(fraud_loss_avg)
        history['train_loss_legitimate'].append(legit_loss_avg)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch

        # Log progress
        logger.info(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Fraud: {fraud_loss_avg:.6f} | "
            f"Legit: {legit_loss_avg:.6f}"
        )

        # Early stopping check
        if early_stopping.step(val_loss):
            logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"\nRestored best model from epoch {history['best_epoch'] + 1}")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)

    return model, history


def train_autoencoder_fixed_epochs(
    model: FeedforwardAutoencoder,
    train_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    fixed_epochs: int = 50
) -> Tuple[FeedforwardAutoencoder, Dict]:
    """
    Train autoencoder for fixed epochs without validation (matches paper's setup).

    No early stopping - trains for exactly 'fixed_epochs' epochs.
    Uses full 70% training data instead of 56%.

    Args:
        model: FeedforwardAutoencoder to train
        train_loader: Training DataLoader (full 70% of data)
        device: cpu/cuda
        config: Training hyperparameters
        fixed_epochs: Number of epochs to train

    Returns:
        (trained_model, history)
    """
    logger.info("=" * 80)
    logger.info("Training Feedforward Autoencoder (FIXED EPOCHS - NO VALIDATION)")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Fixed epochs: {fixed_epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info("NO early stopping - training for fixed epochs")
    logger.info("=" * 80)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],  # Empty, but kept for compatibility
        'train_loss_fraud': [],
        'train_loss_legitimate': [],
        'best_epoch': fixed_epochs - 1  # Last epoch is "best" since no validation
    }

    # Training loop
    for epoch in range(fixed_epochs):
        model.train()
        train_loss_total = 0.0
        train_loss_fraud = 0.0
        train_loss_legit = 0.0
        num_fraud = 0
        num_legit = 0
        num_samples = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(features)
            loss = criterion(reconstructed, features)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # Track overall loss
            batch_size = len(features)
            train_loss_total += loss.item() * batch_size
            num_samples += batch_size

            # Track per-class reconstruction error
            fraud_mask = labels == 1
            legit_mask = labels == 0

            if fraud_mask.sum() > 0:
                with torch.no_grad():
                    fraud_loss = criterion(
                        reconstructed[fraud_mask],
                        features[fraud_mask]
                    )
                    train_loss_fraud += fraud_loss.item() * fraud_mask.sum().item()
                    num_fraud += fraud_mask.sum().item()

            if legit_mask.sum() > 0:
                with torch.no_grad():
                    legit_loss = criterion(
                        reconstructed[legit_mask],
                        features[legit_mask]
                    )
                    train_loss_legit += legit_loss.item() * legit_mask.sum().item()
                    num_legit += legit_mask.sum().item()

        # Compute epoch averages
        train_loss = train_loss_total / num_samples
        fraud_loss_avg = train_loss_fraud / num_fraud if num_fraud > 0 else 0
        legit_loss_avg = train_loss_legit / num_legit if num_legit > 0 else 0

        # Record history
        history['train_loss'].append(train_loss)
        history['train_loss_fraud'].append(fraud_loss_avg)
        history['train_loss_legitimate'].append(legit_loss_avg)

        # Log progress (no validation loss)
        logger.info(
            f"Epoch {epoch + 1:3d}/{fixed_epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Fraud: {fraud_loss_avg:.6f} | "
            f"Legit: {legit_loss_avg:.6f}"
        )

    logger.info("=" * 80)
    logger.info(f"Training complete! Trained for {fixed_epochs} epochs.")
    logger.info("=" * 80)

    return model, history


def train_semi_supervised_autoencoder(
    model: SemiSupervisedAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    classification_lambda: float = 0.01
) -> Tuple[SemiSupervisedAutoencoder, Dict]:
    """
    Train autoencoder with combined reconstruction + classification loss.

    Loss: L = L_reconstruction + λ * L_classification

    This encourages the latent representation to preserve fraud-relevant
    information that pure reconstruction would discard.

    Args:
        model: SemiSupervisedAutoencoder to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: cpu/cuda
        config: Training hyperparameters
        classification_lambda: Weight for classification loss (default: 0.01)

    Returns:
        (trained_model, history)
    """
    logger.info("=" * 80)
    logger.info("Training Semi-Supervised Autoencoder")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Classification lambda: {classification_lambda}")
    logger.info(f"Early stopping patience: {config.patience}")
    logger.info("=" * 80)

    # Losses
    recon_criterion = nn.MSELoss()
    class_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta
    )

    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_class_loss': [],
        'val_loss': [],
        'train_loss_fraud': [],
        'train_loss_legitimate': [],
        'best_epoch': 0
    }

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(config.epochs):
        model.train()
        train_loss_total = 0.0
        train_recon_total = 0.0
        train_class_total = 0.0
        train_loss_fraud = 0.0
        train_loss_legit = 0.0
        num_fraud = 0
        num_legit = 0
        num_samples = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            reconstructed, class_logit = model(features)

            # Reconstruction loss
            recon_loss = recon_criterion(reconstructed, features)

            # Classification loss
            class_loss = class_criterion(class_logit, labels)

            # Combined loss
            loss = recon_loss + classification_lambda * class_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            batch_size = len(features)
            train_loss_total += loss.item() * batch_size
            train_recon_total += recon_loss.item() * batch_size
            train_class_total += class_loss.item() * batch_size
            num_samples += batch_size

            # Track per-class reconstruction error
            fraud_mask = labels == 1
            legit_mask = labels == 0

            if fraud_mask.sum() > 0:
                with torch.no_grad():
                    fraud_loss = recon_criterion(
                        reconstructed[fraud_mask],
                        features[fraud_mask]
                    )
                    train_loss_fraud += fraud_loss.item() * fraud_mask.sum().item()
                    num_fraud += fraud_mask.sum().item()

            if legit_mask.sum() > 0:
                with torch.no_grad():
                    legit_loss = recon_criterion(
                        reconstructed[legit_mask],
                        features[legit_mask]
                    )
                    train_loss_legit += legit_loss.item() * legit_mask.sum().item()
                    num_legit += legit_mask.sum().item()

        # Compute epoch averages
        train_loss = train_loss_total / num_samples
        train_recon = train_recon_total / num_samples
        train_class = train_class_total / num_samples
        fraud_loss_avg = train_loss_fraud / num_fraud if num_fraud > 0 else 0
        legit_loss_avg = train_loss_legit / num_legit if num_legit > 0 else 0

        # Validation
        val_loss = evaluate_autoencoder(model.autoencoder, val_loader, device, recon_criterion)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_recon_loss'].append(train_recon)
        history['train_class_loss'].append(train_class)
        history['val_loss'].append(val_loss)
        history['train_loss_fraud'].append(fraud_loss_avg)
        history['train_loss_legitimate'].append(legit_loss_avg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch

        logger.info(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Loss: {train_loss:.6f} (R:{train_recon:.6f} C:{train_class:.4f}) | "
            f"Val: {val_loss:.6f}"
        )

        if early_stopping.step(val_loss):
            logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"\nRestored best model from epoch {history['best_epoch'] + 1}")

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)

    return model, history


def save_autoencoder_artifacts(
    output_dir: str,
    model: FeedforwardAutoencoder,
    preprocessor: CreditCardPreprocessor,
    history: Dict
) -> None:
    """
    Save trained model and preprocessing artifacts.

    Saves:
    - autoencoder.pt: Model weights and config
    - scaler.pkl: MinMaxScaler fitted on training data
    - training_history.pkl: Loss curves and hyperparameters

    Args:
        output_dir: Directory to save artifacts
        model: Trained autoencoder
        preprocessor: Fitted preprocessor with scaler
        history: Training history
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save autoencoder
    model_path = output_dir / "autoencoder.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
    }, model_path)
    logger.info(f"Saved autoencoder to {model_path}")

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(preprocessor.scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")

    # Save training history
    history_path = output_dir / "training_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"Saved training history to {history_path}")

    logger.info(f"\nAll artifacts saved to {output_dir}/")


def load_autoencoder(
    model_path: str,
    device: torch.device
) -> FeedforwardAutoencoder:
    """
    Load trained autoencoder from checkpoint.

    Args:
        model_path: Path to autoencoder.pt
        device: Device to load model on

    Returns:
        Loaded autoencoder in eval mode
    """
    # Enable safe loading of custom classes
    torch.serialization.add_safe_globals([AutoencoderConfig])

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model = FeedforwardAutoencoder(config=checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Loaded autoencoder from {model_path}")
    logger.info(f"  Parameters: {model.count_parameters():,}")

    return model


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train feedforward autoencoder for credit card fraud detection"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/creditcard.csv",
        help="Path to credit card CSV file"
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=22,
        help="Hidden layer dimension d1 (intermediate size)"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=15,
        help="Latent dimension (bottleneck size)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate (0 = no dropout)"
    )

    # Preprocessing arguments
    parser.add_argument(
        "--no-log-transform",
        action="store_true",
        help="Disable log transform on Amount (normalize raw Amount to [-1,1] instead)"
    )
    parser.add_argument(
        "--no-validation-split",
        action="store_true",
        help="Use full 70%% for training (no validation split, fixed epochs)"
    )
    parser.add_argument(
        "--fixed-epochs",
        type=int,
        default=50,
        help="Number of epochs when training without validation (default: 50)"
    )
    parser.add_argument(
        "--semi-supervised",
        action="store_true",
        help="Use semi-supervised training with classification loss on latent representation"
    )
    parser.add_argument(
        "--classification-lambda",
        type=float,
        default=0.01,
        help="Weight for classification loss in semi-supervised training (default: 0.01)"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/credit_card",
        help="Directory to save trained model and artifacts"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ===== Step 1: Preprocess Data =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Preprocessing Data")
    logger.info("=" * 80)

    # Semi-supervised requires validation split for early stopping
    use_validation = not args.no_validation_split or args.semi_supervised
    if args.semi_supervised and args.no_validation_split:
        logger.warning("Semi-supervised training requires validation split. Ignoring --no-validation-split.")

    preprocessor_config = CreditCardPreprocessorConfig(
        log_transform_amount=not args.no_log_transform,
        use_validation_split=use_validation
    )
    preprocessor = CreditCardPreprocessor(config=preprocessor_config)

    if args.no_log_transform:
        logger.info("Log transform DISABLED - normalizing raw Amount to [-1,1]")
    if args.no_validation_split and not args.semi_supervised:
        logger.info("Validation split DISABLED - using full 70% for training")

    dataloaders, normalized_splits = preprocessor.preprocess(
        filepath=args.data_path,
        batch_size=args.batch_size
    )

    # ===== Step 2: Create Model =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating Model")
    logger.info("=" * 80)

    autoencoder_config = AutoencoderConfig(
        input_dim=30,  # V1-V28 + Hour + Amount_log
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout
    )

    base_autoencoder = FeedforwardAutoencoder(config=autoencoder_config)

    if args.semi_supervised:
        logger.info("Using SEMI-SUPERVISED training with classification loss")
        model = SemiSupervisedAutoencoder(base_autoencoder)
    else:
        model = base_autoencoder

    model.to(device)

    # ===== Step 3: Train Model =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Training Model")
    logger.info("=" * 80)

    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience
    )

    if args.semi_supervised:
        # Semi-supervised training with classification loss
        model, history = train_semi_supervised_autoencoder(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            device=device,
            config=training_config,
            classification_lambda=args.classification_lambda
        )
        # Extract base autoencoder for saving
        model = model.autoencoder
    elif args.no_validation_split:
        # Train for fixed epochs without validation (matches paper)
        model, history = train_autoencoder_fixed_epochs(
            model=model,
            train_loader=dataloaders['train'],
            device=device,
            config=training_config,
            fixed_epochs=args.fixed_epochs
        )
    else:
        # Train with validation and early stopping
        model, history = train_autoencoder(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            device=device,
            config=training_config
        )

    # ===== Step 4: Evaluate on Test Set =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Evaluating on Test Set")
    logger.info("=" * 80)

    criterion = nn.MSELoss()
    test_loss = evaluate_autoencoder(model, dataloaders['test'], device, criterion)
    logger.info(f"Test reconstruction loss: {test_loss:.6f}")

    # ===== Step 5: Save Artifacts =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Saving Artifacts")
    logger.info("=" * 80)

    save_autoencoder_artifacts(
        output_dir=args.output_dir,
        model=model,
        preprocessor=preprocessor,
        history=history
    )

    # ===== Summary =====
    logger.info("\n" + "=" * 80)
    logger.info("Training Summary")
    logger.info("=" * 80)
    logger.info(f"Total epochs: {len(history['train_loss'])}")
    logger.info(f"Best epoch: {history['best_epoch'] + 1}")
    if history['val_loss']:
        logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
    else:
        logger.info("No validation (trained for fixed epochs)")
    logger.info(f"Final test loss: {test_loss:.6f}")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 80)
    logger.info("\n✓ Phase 1 complete! Ready for Phase 2 (MLP classifier training)")


if __name__ == "__main__":
    main()
