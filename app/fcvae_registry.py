"""
FCVAE Model Registry for Multi-Combo Anomaly Detection

Manages 4 independent FCVAE models, one per network/transaction type combination.
Provides unified interface for training, inference, saving, and loading.

Key differences from ModelRegistry (LSTM-AE):
- Loss: ELBO (reconstruction NLL + KLD) instead of MSE
- Augmentation: batch_augment() applied per batch during training
- Scheduler: CosineAnnealingLR instead of no scheduler
- Gradient clipping: value-based (2.0) instead of norm-based (1.0)
"""
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from app.fcvae_model import FCVAE, FCVAEConfig
from app.fcvae_scorer import FCVAEScorer, FCVAEScorerConfig
from app.fcvae_augment import batch_augment, AugmentConfig
from app.transaction_config import COMBO_KEYS

logger = logging.getLogger(__name__)


def combo_to_dirname(combo: Tuple[str, str]) -> str:
    """Convert combo tuple to valid directory name."""
    network, txn_type = combo
    return f"{network}_{txn_type.replace('-', '')}"


def dirname_to_combo(dirname: str) -> Tuple[str, str]:
    """Convert directory name back to combo tuple."""
    parts = dirname.split("_")
    network = parts[0]
    txn_type = "no-pin" if parts[1] == "nopin" else parts[1]
    return (network, txn_type)


class FCVAERegistry:
    """
    Registry managing 4 independent FCVAE models for transaction anomaly detection.

    Provides a unified interface for:
    - Creating and accessing models per combo
    - Training individual or all models (with augmentation)
    - Saving/loading complete model state
    """

    def __init__(
        self,
        model_config: Optional[FCVAEConfig] = None,
        scorer_config: Optional[FCVAEScorerConfig] = None,
        augment_config: Optional[AugmentConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the registry.

        Args:
            model_config: Configuration for FCVAE models
            scorer_config: Configuration for FCVAE scorers
            augment_config: Configuration for data augmentation
            device: Torch device (defaults to CUDA if available)
        """
        self.model_config = model_config or FCVAEConfig()
        self.scorer_config = scorer_config or FCVAEScorerConfig()
        self.augment_config = augment_config or AugmentConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Per-combo storage
        self.models: Dict[Tuple[str, str], FCVAE] = {}
        self.scorers: Dict[Tuple[str, str], FCVAEScorer] = {}
        self.scalers: Dict[Tuple[str, str], StandardScaler] = {}
        self.training_histories: Dict[Tuple[str, str], Dict] = {}

        # Model versioning
        self.model_versions: Dict[Tuple[str, str], int] = {
            combo: 0 for combo in COMBO_KEYS
        }

        logger.info(f"Initialized FCVAERegistry with device={self.device}")
        logger.info(f"  Model config: window={self.model_config.window}, "
                    f"latent_dim={self.model_config.latent_dim}")
        logger.info(f"  Scorer config: mode={self.scorer_config.score_mode}, "
                    f"k={self.scorer_config.hard_criterion_k}")

    def _create_model_for_combo(self, combo: Tuple[str, str]) -> FCVAE:
        """Create a fresh model for a combo."""
        model = FCVAE(config=self.model_config)
        model.to(self.device)
        return model

    def _create_scorer_for_combo(self, combo: Tuple[str, str]) -> FCVAEScorer:
        """Create a fresh scorer for a combo."""
        return FCVAEScorer(config=self.scorer_config)

    def get_model(self, combo: Tuple[str, str]) -> FCVAE:
        """Get the model for a combo, creating if needed."""
        if combo not in self.models:
            self.models[combo] = self._create_model_for_combo(combo)
            logger.debug(f"Created new FCVAE model for {combo}")
        return self.models[combo]

    def get_scorer(self, combo: Tuple[str, str]) -> FCVAEScorer:
        """Get the scorer for a combo, creating if needed."""
        if combo not in self.scorers:
            self.scorers[combo] = self._create_scorer_for_combo(combo)
            logger.debug(f"Created new FCVAEScorer for {combo}")
        return self.scorers[combo]

    def get_scaler(self, combo: Tuple[str, str]) -> Optional[StandardScaler]:
        """Get the scaler for a combo."""
        return self.scalers.get(combo)

    def set_scaler(self, combo: Tuple[str, str], scaler: StandardScaler) -> None:
        """Set the scaler for a combo (from preprocessor)."""
        self.scalers[combo] = scaler

    def train_combo(
        self,
        combo: Tuple[str, str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        learning_rate: float = 5e-4,
        patience: int = 5,
        grad_clip: float = 2.0,
        scheduler_t_max: int = 10,
        use_augmentation: bool = True,
        kl_warmup_epochs: int = 10
    ) -> Dict:
        """
        Train the FCVAE model for a single combo.

        Training loop includes:
        - Data augmentation (point anomalies, segment swaps, missing data)
        - ELBO loss (reconstruction NLL + KLD) with KL annealing
        - CosineAnnealingLR scheduler
        - Gradient clipping by value

        Args:
            combo: (network_type, transaction_type) tuple
            train_loader: DataLoader with training data (x, y, z format)
            val_loader: DataLoader with validation data
            epochs: Maximum training epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            grad_clip: Gradient clipping max value
            scheduler_t_max: CosineAnnealingLR T_max parameter
            use_augmentation: Whether to apply data augmentation
            kl_warmup_epochs: Epochs to linearly ramp KLD weight from 0 to 1

        Returns:
            Training history dict
        """
        model = self.get_model(combo)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "learning_rates": [],
            "kl_weights": [],
        }

        for epoch in range(epochs):
            # Compute KL weight for this epoch (linear warmup)
            # Starts at 0, reaches 1.0 at kl_warmup_epochs
            kl_weight = min(1.0, (epoch + 1) / kl_warmup_epochs) if kl_warmup_epochs > 0 else 1.0

            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                # Unpack batch - expect (x, y, z) from SlidingWindowDataset
                if isinstance(batch, (list, tuple)):
                    x, y, z = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                else:
                    x = batch.to(self.device)
                    y = torch.zeros(x.shape[0], x.shape[2], device=self.device)
                    z = torch.zeros_like(y)

                # Apply data augmentation
                if use_augmentation:
                    x, y, z = batch_augment(x, y, z, self.augment_config)

                optimizer.zero_grad()

                # Forward pass - FCVAE returns loss directly (with KL annealing)
                mu_x, var_x, rec_x, mu, var, loss = model(x, mode="train", kl_weight=kl_weight)

                loss.backward()

                # Gradient clipping by value (not norm)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)

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
                    if isinstance(batch, (list, tuple)):
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)

                    mu_x, var_x, rec_x, mu, var, loss = model(x, mode="valid")
                    val_loss += loss.item()
                    num_val_batches += 1

            val_loss /= num_val_batches

            # Update scheduler
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rates"].append(optimizer.param_groups[0]["lr"])
            history["kl_weights"].append(kl_weight)

            # Check for best model
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                history["best_epoch"] = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging every 5 epochs or at end
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  {combo} Epoch {epoch + 1:3d}: "
                    f"train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.2e}, kl_w={kl_weight:.2f}"
                )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"  {combo}: Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(self.device)

        self.training_histories[combo] = history
        self.model_versions[combo] += 1

        logger.info(
            f"  {combo}: Training complete. Best val loss: {best_val_loss:.6f} "
            f"at epoch {history['best_epoch']}"
        )

        return history

    def fit_scorer(
        self,
        combo: Tuple[str, str],
        val_loader: DataLoader
    ) -> None:
        """
        Fit the scorer for a combo on validation data.

        Args:
            combo: (network_type, transaction_type) tuple
            val_loader: DataLoader with validation data
        """
        model = self.get_model(combo)
        scorer = self.get_scorer(combo)

        # Fit scorer on validation data
        scorer.fit(model, val_loader, self.device)

        # Set threshold using percentile
        point_scores, window_scores = scorer.score_batch(model, val_loader, self.device)
        scorer.set_threshold(point_scores)
        scorer.set_window_threshold(window_scores)

        logger.info(
            f"  {combo}: Scorer fitted. Point threshold: {scorer.point_threshold:.4f}"
        )

    def fit_scorer_with_calibration(
        self,
        combo: Tuple[str, str],
        val_loader: DataLoader,
        preprocessor: "TransactionPreprocessor",
        normal_day_indices: List[int],
        spike_day_index: int,
        dip_day_index: int,
        spike_hours: Tuple[int, int] = (2, 5),
        dip_hours: Tuple[int, int] = (10, 14),
        magnitude_sigma: float = 2.0,
    ) -> Dict:
        """
        Fit scorer with balanced calibration using F1-max optimization.

        Args:
            combo: (network_type, transaction_type) tuple
            val_loader: DataLoader with validation data
            preprocessor: TransactionPreprocessor with window data
            normal_day_indices: Day indices for normal calibration windows
            spike_day_index: Day index to inject spike anomaly
            dip_day_index: Day index to inject dip anomaly
            spike_hours: (start, end) hours for spike injection
            dip_hours: (start, end) hours for dip injection
            magnitude_sigma: Magnitude of anomalies in std units

        Returns:
            Dict with calibration metrics
        """
        from app.synthetic_anomaly import SyntheticAnomalyGenerator, SyntheticAnomalyConfig
        from app.transaction_preprocessor import SlidingWindowDataset

        model = self.get_model(combo)
        scorer = self.get_scorer(combo)

        # Step 1: Fit scorer on validation data
        scorer.fit(model, val_loader, self.device)

        # Step 2: Extract calibration windows using SLIDING windows (not daily)
        # This ensures calibration score distribution matches test score distribution
        window_size = self.model_config.window

        # Get normal sliding windows from threshold-val split
        splits = preprocessor.create_sliding_splits(combo, window_size=window_size, stride=1)
        normalized = preprocessor.normalize_sliding_windows(combo, splits, fit_on="train")

        threshold_data = normalized.get("threshold_val")
        if threshold_data is None or len(threshold_data[0]) == 0:
            raise ValueError(f"No threshold-val data for {combo}")

        normal_windows, normal_labels_array = threshold_data

        # Create anomaly sliding windows by injecting into raw hourly series
        # Then creating sliding windows from the modified series
        hourly_df = preprocessor.combo_hourly[combo]
        hourly_values = hourly_df["count"].values.astype(np.float32)
        hourly_std = np.std(hourly_values)

        # Calculate threshold period boundaries
        hours_per_day = 24
        train_days = preprocessor.config.train_days
        val_days = preprocessor.config.val_days
        threshold_start_hour = (train_days + val_days) * hours_per_day

        # Inject spike anomaly at specified hours within threshold period
        spike_series = hourly_values.copy()
        spike_start = threshold_start_hour + spike_day_index * hours_per_day + spike_hours[0]
        spike_end = threshold_start_hour + spike_day_index * hours_per_day + spike_hours[1]
        spike_series[spike_start:spike_end] += magnitude_sigma * hourly_std

        # Inject dip anomaly at specified hours within threshold period
        dip_series = hourly_values.copy()
        dip_start = threshold_start_hour + dip_day_index * hours_per_day + dip_hours[0]
        dip_end = threshold_start_hour + dip_day_index * hours_per_day + dip_hours[1]
        dip_series[dip_start:dip_end] -= magnitude_sigma * hourly_std
        dip_series[dip_start:dip_end] = np.maximum(0, dip_series[dip_start:dip_end])  # Non-negative

        # Create sliding windows from anomaly series (just for the threshold period)
        threshold_days = preprocessor.config.threshold_days
        threshold_end_hour = threshold_start_hour + threshold_days * hours_per_day

        # Extract windows that start within threshold period
        # Also track which hours in each window actually contain injected anomalies
        anomaly_windows_spike = []
        anomaly_windows_dip = []
        spike_point_labels = []  # Per-point labels: 1 if hour contains spike, 0 otherwise
        dip_point_labels = []    # Per-point labels: 1 if hour contains dip, 0 otherwise

        for start_idx in range(threshold_start_hour, threshold_end_hour - window_size + 1):
            anomaly_windows_spike.append(spike_series[start_idx:start_idx + window_size])
            anomaly_windows_dip.append(dip_series[start_idx:start_idx + window_size])

            # Create point-level labels for this window
            # Mark only the hours that actually contain the injected anomaly
            spike_labels = np.zeros(window_size, dtype=np.float32)
            dip_labels = np.zeros(window_size, dtype=np.float32)

            for offset in range(window_size):
                hour_idx = start_idx + offset
                if spike_start <= hour_idx < spike_end:
                    spike_labels[offset] = 1.0
                if dip_start <= hour_idx < dip_end:
                    dip_labels[offset] = 1.0

            spike_point_labels.append(spike_labels)
            dip_point_labels.append(dip_labels)

        anomaly_windows_spike = np.array(anomaly_windows_spike, dtype=np.float32)
        anomaly_windows_dip = np.array(anomaly_windows_dip, dtype=np.float32)
        spike_point_labels = np.array(spike_point_labels, dtype=np.float32)
        dip_point_labels = np.array(dip_point_labels, dtype=np.float32)

        # Normalize using sliding scaler
        scaler_key = (combo[0], combo[1] + "_sliding")
        if scaler_key not in self.scalers:
            scaler_key = combo  # Fall back to regular scaler

        scaler = self.scalers.get(scaler_key)
        if scaler is None:
            raise ValueError(f"Scaler not found for {combo}")

        # Normal windows are already normalized from normalize_sliding_windows
        normal_norm = normal_windows

        # Normalize anomaly windows
        anomaly_spike_norm = scaler.transform(
            anomaly_windows_spike.flatten().reshape(-1, 1)
        ).reshape(anomaly_windows_spike.shape)
        anomaly_dip_norm = scaler.transform(
            anomaly_windows_dip.flatten().reshape(-1, 1)
        ).reshape(anomaly_windows_dip.shape)

        # Combine anomaly windows and their point-level labels
        anomaly_norm = np.concatenate([anomaly_spike_norm, anomaly_dip_norm], axis=0)
        anomaly_point_labels = np.concatenate([spike_point_labels, dip_point_labels], axis=0)

        # Create DataLoaders (window-level labels for dataset, but we track point labels separately)
        normal_labels = np.zeros_like(normal_norm)
        anomaly_window_labels = np.ones((len(anomaly_norm), window_size))  # Window-level: all 1s

        normal_dataset = SlidingWindowDataset(normal_norm, normal_labels)
        anomaly_dataset = SlidingWindowDataset(anomaly_norm, anomaly_window_labels)

        normal_loader = DataLoader(normal_dataset, batch_size=min(64, len(normal_norm)), shuffle=False)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=min(64, len(anomaly_norm)), shuffle=False)

        # Compute scores
        normal_point_scores, normal_window_scores = scorer.score_batch(
            model, normal_loader, self.device
        )
        anomaly_point_scores, anomaly_window_scores = scorer.score_batch(
            model, anomaly_loader, self.device
        )

        logger.info(f"  {combo}: Sliding window calibration scores:")
        logger.info(f"    Normal windows: {len(normal_window_scores)}, mean={normal_window_scores.mean():.4f}")
        logger.info(f"    Anomaly windows: {len(anomaly_window_scores)}, mean={anomaly_window_scores.mean():.4f}")

        # Step 3: Set thresholds using F1-max
        # CRITICAL: For point-level threshold, only use truly-anomalous points as positives
        # Points in anomaly windows that don't contain the injection are actually normal
        normal_flat = normal_point_scores.flatten()
        anomaly_flat = anomaly_point_scores.flatten()
        anomaly_labels_flat = anomaly_point_labels.flatten()

        # Separate truly-anomalous points from normal-looking points in anomaly windows
        true_anomaly_scores = anomaly_flat[anomaly_labels_flat == 1]
        normal_from_anomaly_windows = anomaly_flat[anomaly_labels_flat == 0]

        # Combine all normal points for threshold calibration
        all_normal_scores = np.concatenate([normal_flat, normal_from_anomaly_windows])

        logger.info(f"    Point-level calibration: {len(all_normal_scores)} normal points, "
                    f"{len(true_anomaly_scores)} truly-anomalous points")

        # Diagnostic: Show score distributions to understand threshold selection
        logger.info(f"    Normal point scores: min={all_normal_scores.min():.4f}, "
                    f"max={all_normal_scores.max():.4f}, mean={all_normal_scores.mean():.4f}, "
                    f"std={all_normal_scores.std():.4f}")
        logger.info(f"    True anomaly scores: min={true_anomaly_scores.min():.4f}, "
                    f"max={true_anomaly_scores.max():.4f}, mean={true_anomaly_scores.mean():.4f}, "
                    f"std={true_anomaly_scores.std():.4f}")
        logger.info(f"    Normal percentiles: 1st={np.percentile(all_normal_scores, 1):.4f}, "
                    f"5th={np.percentile(all_normal_scores, 5):.4f}")
        logger.info(f"    Anomaly percentiles: 95th={np.percentile(true_anomaly_scores, 95):.4f}, "
                    f"99th={np.percentile(true_anomaly_scores, 99):.4f}")
        gap = all_normal_scores.min() - true_anomaly_scores.max()
        logger.info(f"    Separation gap: {gap:.4f} (positive = perfect separation, negative = overlap)")

        optimal_point_threshold, point_metrics = FCVAEScorer.find_optimal_threshold(
            all_normal_scores,
            true_anomaly_scores,
            method="f1_max"
        )
        scorer.point_threshold = optimal_point_threshold

        # Window threshold: use F1-max
        optimal_window_threshold, window_metrics = FCVAEScorer.find_optimal_threshold(
            normal_window_scores,
            anomaly_window_scores,
            method="f1_max"
        )
        scorer.window_threshold = optimal_window_threshold

        logger.info(f"  {combo}: Calibration complete:")
        logger.info(f"    Point threshold (F1-max): {scorer.point_threshold:.4f}")
        logger.info(f"    Point F1 on calibration: {point_metrics.get('f1', 0):.3f}")
        logger.info(f"    Window threshold (F1-max): {optimal_window_threshold:.4f}")
        logger.info(f"    Window F1 on calibration: {window_metrics.get('f1', 0):.3f}")

        # Plot anomaly reconstruction for diagnostic purposes
        try:
            from app.evaluate_fcvae import plot_anomaly_reconstruction
            from pathlib import Path

            output_path = Path("results/fcvae")
            output_path.mkdir(parents=True, exist_ok=True)

            combo_name = f"{combo[0]}_{combo[1]}"

            # Find a window that contains the maximum number of injected spike hours
            spike_label_sums = spike_point_labels.sum(axis=1)
            best_spike_idx = int(np.argmax(spike_label_sums))

            if spike_label_sums[best_spike_idx] > 0:
                # Get corresponding normal window (same index from normal windows)
                # Use the first normal window as baseline
                normal_idx = min(best_spike_idx, len(normal_norm) - 1)

                plot_anomaly_reconstruction(
                    model=model,
                    normal_window=normal_norm[normal_idx],
                    anomaly_window=anomaly_spike_norm[best_spike_idx],
                    anomaly_labels=spike_point_labels[best_spike_idx],
                    point_threshold=scorer.point_threshold,
                    device=self.device,
                    output_path=output_path,
                    combo_name=combo_name,
                    anomaly_type="spike",
                )

            # Also plot dip reconstruction
            dip_label_sums = dip_point_labels.sum(axis=1)
            best_dip_idx = int(np.argmax(dip_label_sums))

            if dip_label_sums[best_dip_idx] > 0:
                normal_idx = min(best_dip_idx, len(normal_norm) - 1)

                plot_anomaly_reconstruction(
                    model=model,
                    normal_window=normal_norm[normal_idx],
                    anomaly_window=anomaly_dip_norm[best_dip_idx],
                    anomaly_labels=dip_point_labels[best_dip_idx],
                    point_threshold=scorer.point_threshold,
                    device=self.device,
                    output_path=output_path,
                    combo_name=combo_name,
                    anomaly_type="dip",
                )

        except Exception as e:
            logger.warning(f"Failed to plot anomaly reconstruction for {combo}: {e}")

        return {
            "normal_window_scores": normal_window_scores.tolist(),
            "anomaly_window_scores": anomaly_window_scores.tolist(),
            "normal_point_scores": normal_point_scores.flatten().tolist(),
            "anomaly_point_scores": anomaly_point_scores.flatten().tolist(),
            "point_threshold": float(scorer.point_threshold),
            "point_metrics": point_metrics,
            "window_threshold": optimal_window_threshold,
            "window_metrics": window_metrics,
        }

    def predict(
        self,
        combo: Tuple[str, str],
        data_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run prediction for a combo.

        Args:
            combo: (network_type, transaction_type) tuple
            data_loader: DataLoader with data to predict

        Returns:
            Tuple of (window_predictions, window_scores, point_scores)
        """
        model = self.get_model(combo)
        scorer = self.get_scorer(combo)

        model.eval()

        point_scores, window_scores = scorer.score_batch(model, data_loader, self.device)
        point_predictions = scorer.predict_points(point_scores)
        window_predictions = scorer.predict_windows_from_points(point_predictions)

        return window_predictions, window_scores, point_scores

    def save_all(self, output_dir: str) -> None:
        """
        Save all models, scorers, and scalers.

        Directory structure:
        output_dir/
            Accel_CMP/
                model.pt
                scorer.pkl
                scaler.pkl
                history.pkl
            ...
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for combo in COMBO_KEYS:
            combo_dir = output_dir / combo_to_dirname(combo)
            combo_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            if combo in self.models:
                model = self.models[combo]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config,
                    "model_version": self.model_versions.get(combo, 0),
                }, combo_dir / "model.pt")

            # Save scorer
            if combo in self.scorers:
                self.scorers[combo].save(combo_dir / "scorer.pkl")

            # Save scaler (check both regular and sliding keys)
            scaler_key = (combo[0], combo[1] + "_sliding")
            if scaler_key in self.scalers:
                with open(combo_dir / "scaler.pkl", "wb") as f:
                    pickle.dump(self.scalers[scaler_key], f)
            elif combo in self.scalers:
                with open(combo_dir / "scaler.pkl", "wb") as f:
                    pickle.dump(self.scalers[combo], f)

            # Save training history
            if combo in self.training_histories:
                with open(combo_dir / "history.pkl", "wb") as f:
                    pickle.dump(self.training_histories[combo], f)

            logger.debug(f"Saved artifacts for {combo}")

        # Save registry config
        with open(output_dir / "registry_config.pkl", "wb") as f:
            pickle.dump({
                "model_config": self.model_config,
                "scorer_config": self.scorer_config,
                "augment_config": self.augment_config,
                "model_versions": self.model_versions,
            }, f)

        logger.info(f"Saved all FCVAE models to {output_dir}")

    def load_all(self, model_dir: str) -> None:
        """
        Load all models, scorers, and scalers.

        Args:
            model_dir: Base directory containing saved models
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Load registry config if exists
        config_path = model_dir / "registry_config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = pickle.load(f)
                self.model_config = config.get("model_config", self.model_config)
                self.scorer_config = config.get("scorer_config", self.scorer_config)
                self.augment_config = config.get("augment_config", self.augment_config)
                self.model_versions = config.get("model_versions", self.model_versions)

        # Allow FCVAEConfig for safe loading
        torch.serialization.add_safe_globals([FCVAEConfig])

        for combo in COMBO_KEYS:
            combo_dir = model_dir / combo_to_dirname(combo)
            if not combo_dir.exists():
                logger.warning(f"No saved model for {combo}")
                continue

            # Load model
            model_path = combo_dir / "model.pt"
            if model_path.exists():
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )
                model = FCVAE(config=checkpoint["model_config"])
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()
                self.models[combo] = model

                if "model_version" in checkpoint:
                    self.model_versions[combo] = checkpoint["model_version"]

            # Load scorer
            scorer_path = combo_dir / "scorer.pkl"
            if scorer_path.exists():
                self.scorers[combo] = FCVAEScorer.load(scorer_path)

            # Load scaler
            scaler_path = combo_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    # Store with sliding suffix for consistency
                    scaler_key = (combo[0], combo[1] + "_sliding")
                    self.scalers[scaler_key] = pickle.load(f)

            # Load history
            history_path = combo_dir / "history.pkl"
            if history_path.exists():
                with open(history_path, "rb") as f:
                    self.training_histories[combo] = pickle.load(f)

            logger.debug(f"Loaded artifacts for {combo}")

        logger.info(f"Loaded FCVAE models from {model_dir}")

    def get_stats(self) -> Dict:
        """Get registry statistics."""
        stats = {
            "device": str(self.device),
            "model_config": {
                "window": self.model_config.window,
                "latent_dim": self.model_config.latent_dim,
                "condition_emb_dim": self.model_config.condition_emb_dim,
            },
            "scorer_config": {
                "score_mode": self.scorer_config.score_mode,
                "hard_criterion_k": self.scorer_config.hard_criterion_k,
            },
            "augment_config": {
                "point_ano_rate": self.augment_config.point_ano_rate,
                "seg_ano_rate": self.augment_config.seg_ano_rate,
                "missing_data_rate": self.augment_config.missing_data_rate,
            },
            "combos": {},
        }

        for combo in COMBO_KEYS:
            combo_stats = {
                "has_model": combo in self.models,
                "has_scorer": combo in self.scorers,
                "has_scaler": any(k[0] == combo[0] and combo[1] in k[1] for k in self.scalers.keys()) if self.scalers else False,
                "model_version": self.model_versions.get(combo, 0),
            }

            if combo in self.scorers and self.scorers[combo].is_fitted:
                scorer = self.scorers[combo]
                combo_stats["point_threshold"] = scorer.point_threshold
                combo_stats["window_threshold"] = scorer.window_threshold

            if combo in self.training_histories:
                history = self.training_histories[combo]
                combo_stats["best_epoch"] = history.get("best_epoch", 0)
                combo_stats["final_val_loss"] = history["val_loss"][-1] if history["val_loss"] else None

            stats["combos"][f"{combo[0]}_{combo[1]}"] = combo_stats

        return stats
