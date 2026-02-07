"""
Training script for MLP classifier on autoencoder latent features.

This implements Phase 1, Steps 3-4:
- Step 3: Extract latent features using frozen encoder
- Step 4: Train MLP classifier on latent features
"""

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from credit_card_preprocessor import CreditCardPreprocessor, CreditCardPreprocessorConfig
from feedforward_autoencoder import AutoencoderConfig, FeedforwardAutoencoder
from mlp_classifier import MLPClassifier, MLPConfig

logger = logging.getLogger(__name__)


@dataclass
class ClassifierTrainingConfig:
    """Configuration for MLP classifier training."""

    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 256
    patience: int = 15               # Higher patience than autoencoder
    min_delta: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    class_weight_fraud: float = 1.0  # Weight for fraud class (default: no weighting)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
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


def extract_latent_features(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract latent features using frozen encoder.

    Args:
        encoder: Frozen encoder network
        dataloader: DataLoader with (features, labels)
        device: cpu/cuda

    Returns:
        (latent_features, labels) as numpy arrays
    """
    encoder.eval()
    all_latent = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)

            # Extract latent representation
            latent = encoder(features)

            all_latent.append(latent.cpu().numpy())
            all_labels.append(labels.numpy())

    latent_features = np.vstack(all_latent)
    labels = np.concatenate(all_labels)

    return latent_features, labels


def extract_recon_error_features(
    autoencoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract original features with appended reconstruction error.

    Instead of using latent features (15 dims), this uses original features (30 dims)
    plus the scalar reconstruction error, giving 31 features total.

    This approach cannot do worse than using original features alone since the
    autoencoder ADDS information rather than replacing it.

    Args:
        autoencoder: Trained autoencoder (full model, not just encoder)
        dataloader: DataLoader with (features, labels)
        device: cpu/cuda

    Returns:
        (features_with_recon_error, labels) as numpy arrays
        features_with_recon_error shape: (N, 31)
    """
    autoencoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)

            # Get reconstruction
            reconstructed = autoencoder(features)

            # Compute per-sample reconstruction error (L2 norm)
            recon_error = torch.norm(features - reconstructed, dim=1, keepdim=True)

            # Concatenate: [original features (30), recon_error (1)] = 31 features
            features_with_error = torch.cat([features, recon_error], dim=1)

            all_features.append(features_with_error.cpu().numpy())
            all_labels.append(labels.numpy())

    combined_features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return combined_features, labels


def train_classifier(
    model: MLPClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: ClassifierTrainingConfig
) -> Tuple[MLPClassifier, Dict]:
    """
    Train MLP classifier on latent features.

    Args:
        model: MLPClassifier to train
        train_loader: Training DataLoader with (latent_features, labels)
        val_loader: Validation DataLoader
        device: cpu/cuda
        config: Training configuration

    Returns:
        (trained_model, history)
    """
    logger.info("=" * 80)
    logger.info("Training MLP Classifier")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Early stopping patience: {config.patience}")
    logger.info(f"Fraud class weight: {config.class_weight_fraud}")
    logger.info("=" * 80)

    # Loss with class weighting
    if config.class_weight_fraud != 1.0:
        # pos_weight for BCEWithLogitsLoss (applied to positive class)
        # We need to use BCEWithLogitsLoss instead of BCELoss for pos_weight
        # But our model outputs probabilities, so we'll use weighted BCE manually
        criterion = nn.BCELoss(reduction='none')
        use_class_weight = True
    else:
        criterion = nn.BCELoss()
        use_class_weight = False

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'best_epoch': 0
    }

    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(config.epochs):
        # ===== Training Phase =====
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []

        for latent, labels in train_loader:
            latent, labels = latent.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            probs = model(latent).squeeze()

            # Compute loss
            if use_class_weight:
                # Apply class weights
                weights = torch.ones_like(labels)
                weights[labels == 1] = config.class_weight_fraud
                loss = (criterion(probs, labels) * weights).mean()
            else:
                loss = criterion(probs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # Track metrics
            train_loss += loss.item() * len(latent)
            preds = (probs.detach() >= 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Compute epoch metrics
        train_loss /= len(train_loader.dataset)
        train_precision = precision_score(all_labels, all_preds, zero_division=0)
        train_recall = recall_score(all_labels, all_preds, zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)

        # ===== Validation Phase =====
        val_loss, val_precision, val_recall, val_f1 = evaluate_classifier(
            model, val_loader, device, criterion, use_class_weight, config.class_weight_fraud
        )

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch

        # Log progress
        logger.info(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val P/R/F1: {val_precision:.4f}/{val_recall:.4f}/{val_f1:.4f}"
        )

        # Early stopping
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


def evaluate_classifier(
    model: MLPClassifier,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_class_weight: bool = False,
    class_weight_fraud: float = 1.0
) -> Tuple[float, float, float, float]:
    """
    Evaluate classifier.

    Args:
        model: Trained classifier
        dataloader: DataLoader
        device: cpu/cuda
        criterion: Loss function
        use_class_weight: Whether to use class weighting
        class_weight_fraud: Weight for fraud class

    Returns:
        (loss, precision, recall, f1)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for latent, labels in dataloader:
            latent, labels = latent.to(device), labels.to(device)

            # Forward pass
            probs = model(latent).squeeze()

            # Compute loss
            if use_class_weight:
                weights = torch.ones_like(labels)
                weights[labels == 1] = class_weight_fraud
                loss = (criterion(probs, labels) * weights).mean()
            else:
                loss = criterion(probs, labels)

            total_loss += loss.item() * len(latent)

            # Predictions
            preds = (probs >= 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(dataloader.dataset)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, precision, recall, f1


def train_classifier_fixed_epochs(
    model: MLPClassifier,
    train_loader: DataLoader,
    device: torch.device,
    config: ClassifierTrainingConfig,
    fixed_epochs: int = 50
) -> Tuple[MLPClassifier, Dict]:
    """
    Train MLP classifier for fixed epochs without validation (matches paper's setup).

    No early stopping - trains for exactly 'fixed_epochs' epochs.

    Args:
        model: MLPClassifier to train
        train_loader: Training DataLoader (full 70% of data)
        device: cpu/cuda
        config: Training configuration
        fixed_epochs: Number of epochs to train

    Returns:
        (trained_model, history)
    """
    logger.info("=" * 80)
    logger.info("Training MLP Classifier (FIXED EPOCHS - NO VALIDATION)")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Fixed epochs: {fixed_epochs}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Fraud class weight: {config.class_weight_fraud}")
    logger.info("NO early stopping - training for fixed epochs")
    logger.info("=" * 80)

    # Loss with class weighting
    if config.class_weight_fraud != 1.0:
        criterion = nn.BCELoss(reduction='none')
        use_class_weight = True
    else:
        criterion = nn.BCELoss()
        use_class_weight = False

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],  # Empty, kept for compatibility
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'best_epoch': fixed_epochs - 1
    }

    # Training loop
    for epoch in range(fixed_epochs):
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []

        for latent, labels in train_loader:
            latent, labels = latent.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            probs = model(latent).squeeze()

            # Compute loss
            if use_class_weight:
                weights = torch.ones_like(labels)
                weights[labels == 1] = config.class_weight_fraud
                loss = (criterion(probs, labels) * weights).mean()
            else:
                loss = criterion(probs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # Track metrics
            train_loss += loss.item() * len(latent)
            preds = (probs.detach() >= 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Compute epoch metrics
        train_loss /= len(train_loader.dataset)
        train_precision = precision_score(all_labels, all_preds, zero_division=0)
        train_recall = recall_score(all_labels, all_preds, zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)

        # Log progress (no validation)
        logger.info(
            f"Epoch {epoch + 1:3d}/{fixed_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"P/R/F1: {train_precision:.4f}/{train_recall:.4f}/{train_f1:.4f}"
        )

    logger.info("=" * 80)
    logger.info(f"Training complete! Trained for {fixed_epochs} epochs.")
    logger.info("=" * 80)

    return model, history


def evaluate_detailed(
    model: MLPClassifier,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str = "Test"
) -> Dict:
    """
    Detailed evaluation with classification report.

    Args:
        model: Trained classifier
        dataloader: DataLoader
        device: cpu/cuda
        split_name: Name of split for logging

    Returns:
        Dict with detailed metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for latent, labels in dataloader:
            latent = latent.to(device)

            probs = model(latent).squeeze()
            preds = (probs >= 0.5).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"{split_name} Set Evaluation")
    logger.info(f"{'=' * 80}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(
        all_labels, all_preds,
        target_names=['Legitimate', 'Fraud'],
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("\nConfusion Matrix:")
    logger.info(f"                 Predicted")
    logger.info(f"               Legit  Fraud")
    logger.info(f"Actual Legit   {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    logger.info(f"       Fraud   {cm[1, 0]:5d}  {cm[1, 1]:5d}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train MLP classifier on autoencoder latent features"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/creditcard.csv",
        help="Path to credit card CSV file"
    )
    parser.add_argument(
        "--autoencoder-path",
        type=str,
        default="models/credit_card/autoencoder.pt",
        help="Path to trained autoencoder"
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="models/credit_card/scaler.pkl",
        help="Path to fitted scaler"
    )

    # Preprocessing arguments
    parser.add_argument(
        "--no-log-transform",
        action="store_true",
        help="Disable log transform on Amount (must match autoencoder training)"
    )
    parser.add_argument(
        "--use-recon-error",
        action="store_true",
        help="Use original features + reconstruction error (31 dims) instead of latent features (15 dims)"
    )
    parser.add_argument(
        "--raw-features",
        action="store_true",
        help="Use raw 30 features directly (M2 model - no autoencoder needed)"
    )
    parser.add_argument(
        "--no-validation-split",
        action="store_true",
        help="Use full 70%% for training (no validation split, fixed epochs)"
    )
    parser.add_argument(
        "--fixed-epochs",
        type=int,
        default=75,
        help="Number of epochs when training without validation (default: 50)"
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
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--class-weight-fraud",
        type=float,
        default=1.0,
        help="Weight for fraud class (default: 1.0, no weighting)"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/credit_card",
        help="Directory to save trained classifier"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix for output files (e.g., '_m2' for mlp_classifier_m2.pt)"
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

    preprocessor_config = CreditCardPreprocessorConfig(
        log_transform_amount=not args.no_log_transform,
        use_validation_split=not args.no_validation_split
    )
    preprocessor = CreditCardPreprocessor(config=preprocessor_config)

    if args.no_log_transform:
        logger.info("Log transform DISABLED - using raw Amount normalized to [-1,1]")
    if args.no_validation_split:
        logger.info("Validation split DISABLED - using full 70% for training")

    dataloaders, normalized_splits = preprocessor.preprocess(
        filepath=args.data_path,
        batch_size=args.batch_size
    )

    # Determine which splits to process
    split_names = ['train', 'test'] if args.no_validation_split else ['train', 'val', 'test']

    # ===== Step 2: Prepare Features =====
    if args.raw_features:
        # M2 model: Use raw features directly, no autoencoder needed
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Using Raw Features (M2 model - no autoencoder)")
        logger.info("=" * 80)

        extracted_features = {}
        labels = {}
        for split_name in split_names:
            X, y = normalized_splits[split_name]
            extracted_features[split_name] = X
            labels[split_name] = y
            logger.info(f"  {split_name}: {X.shape[0]:,} samples, shape {X.shape}")

        input_dim = extracted_features['train'].shape[1]
        logger.info(f"MLP input dimension: {input_dim}")
        autoencoder = None  # No autoencoder for raw features

    else:
        # Load autoencoder and extract features
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Loading Trained Autoencoder")
        logger.info("=" * 80)

        # Enable safe loading
        torch.serialization.add_safe_globals([AutoencoderConfig])

        checkpoint = torch.load(args.autoencoder_path, map_location=device, weights_only=True)
        autoencoder = FeedforwardAutoencoder(config=checkpoint['model_config'])
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder.to(device)
        autoencoder.eval()

        logger.info(f"Loaded autoencoder from {args.autoencoder_path}")
        logger.info(f"  Parameters: {autoencoder.count_parameters():,}")

        # Freeze encoder
        for param in autoencoder.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

        # ===== Step 3: Extract Features =====
        logger.info("\n" + "=" * 80)
        if args.use_recon_error:
            logger.info("Step 3: Extracting Original Features + Reconstruction Error (31 dims)")
        else:
            logger.info("Step 3: Extracting Latent Features (15 dims)")
        logger.info("=" * 80)

        extracted_features = {}
        labels = {}

        for split_name in split_names:
            logger.info(f"Extracting {split_name} features...")
            if args.use_recon_error:
                # Use original features + reconstruction error (31 dims)
                feats, lbls = extract_recon_error_features(
                    autoencoder,  # Full autoencoder, not just encoder
                    dataloaders[split_name],
                    device
                )
            else:
                # Use latent features (15 dims)
                feats, lbls = extract_latent_features(
                    autoencoder.encoder,
                    dataloaders[split_name],
                    device
                )
            extracted_features[split_name] = feats
            labels[split_name] = lbls
            logger.info(f"  {split_name}: {feats.shape[0]:,} samples, shape {feats.shape}")

        # Determine input dimension based on feature extraction method
        input_dim = extracted_features['train'].shape[1]
        logger.info(f"MLP input dimension: {input_dim}")

    # Create new DataLoaders with extracted features
    feature_dataloaders = {}
    for split_name in split_names:
        dataset = TensorDataset(
            torch.FloatTensor(extracted_features[split_name]),
            torch.FloatTensor(labels[split_name])
        )
        shuffle = (split_name == 'train')
        feature_dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle
        )

    # ===== Step 4: Train MLP Classifier =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Training MLP Classifier")
    logger.info("=" * 80)

    # Adjust MLP architecture based on input dimension
    if args.raw_features:
        # 30 input features (raw): M2 model architecture
        mlp_config = MLPConfig(
            input_dim=30,
            hidden_dim1=22,  # Scaled for 30 inputs
            hidden_dim2=12
        )
    elif args.use_recon_error:
        # 31 input features: scale hidden layers proportionally
        mlp_config = MLPConfig(
            input_dim=31,
            hidden_dim1=20,  # Scaled up from 13
            hidden_dim2=10   # Scaled up from 7
        )
    else:
        # 15 input features (latent)
        mlp_config = MLPConfig(
            input_dim=15,
            hidden_dim1=13,
            hidden_dim2=7
        )
    classifier = MLPClassifier(config=mlp_config)
    classifier.to(device)

    training_config = ClassifierTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        class_weight_fraud=args.class_weight_fraud
    )

    if args.no_validation_split:
        # Train for fixed epochs without validation (matches paper)
        classifier, history = train_classifier_fixed_epochs(
            model=classifier,
            train_loader=feature_dataloaders['train'],
            device=device,
            config=training_config,
            fixed_epochs=args.fixed_epochs
        )
    else:
        # Train with validation and early stopping
        classifier, history = train_classifier(
            model=classifier,
            train_loader=feature_dataloaders['train'],
            val_loader=feature_dataloaders['val'],
            device=device,
            config=training_config
        )

    # ===== Step 5: Evaluate on Test Set =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Evaluating on Test Set")
    logger.info("=" * 80)

    test_results = evaluate_detailed(
        classifier,
        feature_dataloaders['test'],
        device,
        split_name="Test"
    )

    # ===== Step 6: Save Artifacts =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Saving Artifacts")
    logger.info("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.output_suffix

    # Save classifier
    classifier_path = output_dir / f"mlp_classifier{suffix}.pt"
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'model_config': classifier.config,
    }, classifier_path)
    logger.info(f"Saved classifier to {classifier_path}")

    # Save training history
    history_path = output_dir / f"mlp_training_history{suffix}.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"Saved training history to {history_path}")

    # Save test results
    results_path = output_dir / f"test_results{suffix}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(test_results, f)
    logger.info(f"Saved test results to {results_path}")

    # ===== Summary =====
    logger.info("\n" + "=" * 80)
    logger.info("Training Summary")
    logger.info("=" * 80)
    if autoencoder is not None:
        logger.info(f"Autoencoder: {autoencoder.count_parameters():,} params")
        logger.info(f"Classifier: {classifier.count_parameters():,} params")
        logger.info(f"Total M1 model: {autoencoder.count_parameters() + classifier.count_parameters():,} params")
    else:
        logger.info(f"Model type: M2 (raw features, no autoencoder)")
        logger.info(f"Classifier: {classifier.count_parameters():,} params")
    logger.info(f"\nBest epoch: {history['best_epoch'] + 1}")
    if history['val_f1']:
        logger.info(f"Best validation F1: {max(history['val_f1']):.4f}")
    else:
        logger.info("No validation (trained for fixed epochs)")
    logger.info(f"\nTest Metrics:")
    logger.info(f"  Precision: {test_results['precision']:.4f}")
    logger.info(f"  Recall: {test_results['recall']:.4f}")
    logger.info(f"  F1 Score: {test_results['f1']:.4f}")
    logger.info("=" * 80)
    if autoencoder is not None:
        logger.info("\nPhase 1 (M1 Model) complete!")
    else:
        logger.info("\nM2 Model (raw features) complete!")


if __name__ == "__main__":
    main()
