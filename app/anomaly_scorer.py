"""
Anomaly Scorer for LSTM Encoder-Decoder

Implements the scoring methodology from Malhotra et al. (2016):
1. Fit error distribution (μ, Σ) on normal training data
2. Compute Mahalanobis distance as anomaly score
3. Set threshold using validation data
4. Predict anomalies based on threshold

Uses the full covariance matrix to capture dependencies
between reconstruction errors at different timesteps.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """Configuration for the anomaly scorer."""
    threshold_method: str = "percentile"  # "percentile" or "sigma"
    threshold_percentile: float = 95.0    # For percentile method
    threshold_sigma: float = 3.0          # For sigma method (k * std)
    regularization: float = 1e-6          # Regularization for covariance inversion
    # Point-level scoring parameters (Malhotra et al. 2016)
    scoring_mode: str = "point"           # "point" (paper) or "window" (legacy)
    hard_criterion_k: int = 5             # Points exceeding threshold to flag window as anomalous


class AnomalyScorer:
    """
    Computes anomaly scores using Mahalanobis distance.

    The scorer learns the distribution of reconstruction errors on
    normal (training) data, fitting a multivariate Gaussian with
    mean μ and covariance Σ.

    Anomaly Score (Mahalanobis distance):
        a(i) = (e(i) - μ)ᵀ Σ⁻¹ (e(i) - μ)

    Where:
        - e(i) is the reconstruction error vector for sequence i
        - μ is the mean error vector (learned from training)
        - Σ is the covariance matrix of errors (learned from training)

    Attributes:
        mu: Mean reconstruction error vector, shape (seq_len,)
        cov: Covariance matrix of errors, shape (seq_len, seq_len)
        cov_inv: Inverse covariance matrix (precision matrix)
        threshold: Anomaly score threshold for classification
    """

    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()
        # Window-level attributes (legacy)
        self.mu: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        # Point-level attributes (Malhotra et al. 2016)
        self.mu_point: Optional[np.ndarray] = None      # Mean error across all points, shape (m,)
        self.sigma_point: Optional[np.ndarray] = None   # Variance (for univariate), shape (m,)
        self.sigma_inv_point: Optional[np.ndarray] = None  # Inverse variance
        self.cov_point: Optional[np.ndarray] = None     # Covariance across features (multivariate)
        self.cov_inv_point: Optional[np.ndarray] = None # Inverse covariance
        self.point_threshold: Optional[float] = None    # Per-point threshold τ
        self.is_fitted: bool = False

        logger.info(f"Initialized AnomalyScorer with config: {self.config}")

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        device: torch.device
    ) -> None:
        """
        Fit the error distribution on training (normal) data.

        Supports two modes:
        - "point" (Malhotra et al. 2016): Pool all points across windows,
          compute μ and σ across features (scalar for univariate)
        - "window" (legacy): Compute μ and Σ across time axis

        Args:
            model: Trained LSTM encoder-decoder model
            train_loader: DataLoader with normal training sequences
            device: Device to run inference on (cpu/cuda)
        """
        model.eval()
        all_errors = []

        logger.info(f"Fitting error distribution on training data (mode={self.config.scoring_mode})...")

        with torch.no_grad():
            for batch in train_loader:
                x = batch.to(device)
                x_reconstructed = model(x)

                # Point-wise absolute error: (batch, seq_len, m)
                errors = torch.abs(x - x_reconstructed)
                all_errors.append(errors.cpu().numpy())

        # Concatenate all batches: (num_sequences, seq_len, m)
        all_errors = np.concatenate(all_errors, axis=0)
        num_sequences, seq_len, num_features = all_errors.shape

        if self.config.scoring_mode == "point":
            # Point-level scoring (Malhotra et al. 2016)
            # Pool all reconstruction errors across all time points and windows
            # Shape: (num_sequences * seq_len, m)
            all_errors_pooled = all_errors.reshape(-1, num_features)

            # μ shape: (m,) - mean error across all normal points
            self.mu_point = np.mean(all_errors_pooled, axis=0)

            if num_features == 1:
                # Univariate case: compute scalar variance
                self.sigma_point = np.var(all_errors_pooled, axis=0) + self.config.regularization
                self.sigma_inv_point = 1.0 / self.sigma_point
                logger.info(f"Fitted point-level scorer (univariate):")
                logger.info(f"  mu_point: {self.mu_point[0]:.6f}")
                logger.info(f"  sigma_point: {self.sigma_point[0]:.6f}")
            else:
                # Multivariate case: compute full covariance
                errors_centered = all_errors_pooled - self.mu_point
                self.cov_point = np.cov(errors_centered, rowvar=False)
                self.cov_point += self.config.regularization * np.eye(num_features)
                try:
                    self.cov_inv_point = np.linalg.inv(self.cov_point)
                except np.linalg.LinAlgError:
                    logger.warning("Point-level covariance singular, using pseudo-inverse")
                    self.cov_inv_point = np.linalg.pinv(self.cov_point)
                logger.info(f"Fitted point-level scorer (multivariate, m={num_features}):")
                logger.info(f"  mu_point shape: {self.mu_point.shape}")
                logger.info(f"  cov_point shape: {self.cov_point.shape}")

            # Also fit window-level for backward compatibility
            all_errors_squeezed = all_errors.squeeze(-1)  # (num_sequences, seq_len)
            self.mu = np.mean(all_errors_squeezed, axis=0)
            errors_centered = all_errors_squeezed - self.mu
            self.cov = np.cov(errors_centered, rowvar=False)
            self.cov += self.config.regularization * np.eye(seq_len)
            try:
                self.cov_inv = np.linalg.inv(self.cov)
            except np.linalg.LinAlgError:
                self.cov_inv = np.linalg.pinv(self.cov)

        else:
            # Window-level scoring (legacy)
            all_errors = all_errors.squeeze(-1)  # (num_sequences, seq_len)
            self.mu = np.mean(all_errors, axis=0)  # (seq_len,)
            errors_centered = all_errors - self.mu
            self.cov = np.cov(errors_centered, rowvar=False)
            self.cov += self.config.regularization * np.eye(seq_len)
            try:
                self.cov_inv = np.linalg.inv(self.cov)
            except np.linalg.LinAlgError:
                logger.warning("Covariance matrix singular, using pseudo-inverse")
                self.cov_inv = np.linalg.pinv(self.cov)

            logger.info(f"Fitted window-level scorer:")
            logger.info(f"  Mean error range: [{self.mu.min():.4f}, {self.mu.max():.4f}]")
            logger.info(f"  Covariance matrix shape: {self.cov.shape}")
            logger.info(f"  Covariance condition number: {np.linalg.cond(self.cov):.2e}")

        self.is_fitted = True
        logger.info(f"Fitted on {num_sequences} sequences of length {seq_len}")

    def _mahalanobis_distance(self, error: np.ndarray) -> float:
        """
        Compute Mahalanobis distance for a single error vector.

        d² = (e - μ)ᵀ Σ⁻¹ (e - μ)

        Args:
            error: Error vector, shape (seq_len,)

        Returns:
            Squared Mahalanobis distance (anomaly score)
        """
        diff = error - self.mu  # (seq_len,)
        # d² = diff @ Σ⁻¹ @ diff
        return diff @ self.cov_inv @ diff

    def compute_scores(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for sequences using Mahalanobis distance.

        Args:
            model: Trained LSTM encoder-decoder model
            data_loader: DataLoader with sequences to score
            device: Device to run inference on

        Returns:
            Tuple of:
                - sequence_scores: Mahalanobis distance per sequence, shape (num_sequences,)
                - reconstruction_errors: Raw errors, shape (num_sequences, seq_len)
        """
        if not self.is_fitted:
            raise ValueError("Scorer not fitted. Call fit() first.")

        model.eval()
        all_scores = []
        all_errors = []

        with torch.no_grad():
            for batch in data_loader:
                x = batch.to(device)
                x_reconstructed = model(x)

                # Point-wise absolute error: (batch, seq_len)
                errors = torch.abs(x - x_reconstructed).cpu().numpy().squeeze(-1)

                # Compute Mahalanobis distance for each sequence in batch
                for error in errors:
                    score = self._mahalanobis_distance(error)
                    all_scores.append(score)

                all_errors.append(errors)

        return np.array(all_scores), np.concatenate(all_errors, axis=0)

    def compute_point_scores(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-point anomaly scores for all sequences.

        Implements Malhotra et al. (2016) point-level scoring:
        a^(i) = (e^(i) - μ)ᵀ Σ⁻¹ (e^(i) - μ)

        For univariate data (m=1), this simplifies to:
        a^(i) = ((e^(i) - μ) / σ)²

        Args:
            model: Trained LSTM encoder-decoder model
            data_loader: DataLoader with sequences to score
            device: Device to run inference on

        Returns:
            Tuple of:
                - point_scores: Per-point anomaly scores, shape (num_sequences, seq_len)
                - window_scores: Max point score per window, shape (num_sequences,)
                - reconstruction_errors: Raw errors, shape (num_sequences, seq_len, m)
        """
        if not self.is_fitted:
            raise ValueError("Scorer not fitted. Call fit() first.")

        if self.mu_point is None:
            raise ValueError("Point-level parameters not fitted. Ensure scoring_mode='point' during fit().")

        model.eval()
        all_point_scores = []
        all_errors = []

        with torch.no_grad():
            for batch in data_loader:
                x = batch.to(device)
                x_reconstructed = model(x)

                # Point-wise absolute error: (batch, seq_len, m)
                errors = torch.abs(x - x_reconstructed).cpu().numpy()

                # Compute point-level Mahalanobis distance
                if errors.shape[-1] == 1:
                    # Univariate: (e - μ)² / σ²
                    errors_squeezed = errors.squeeze(-1)  # (batch, seq_len)
                    point_scores = ((errors_squeezed - self.mu_point[0]) ** 2) / self.sigma_point[0]
                else:
                    # Multivariate: full Mahalanobis
                    diff = errors - self.mu_point  # (batch, seq_len, m)
                    # Compute (diff @ cov_inv @ diff) for each point
                    point_scores = np.einsum('ijk,kl,ijl->ij', diff, self.cov_inv_point, diff)

                all_point_scores.append(point_scores)
                all_errors.append(errors)

        point_scores = np.concatenate(all_point_scores, axis=0)  # (num_sequences, seq_len)
        errors = np.concatenate(all_errors, axis=0)  # (num_sequences, seq_len, m)

        # Window score: max point score in window (Li thesis approach for AUC)
        window_scores = np.max(point_scores, axis=1)  # (num_sequences,)

        return point_scores, window_scores, errors

    def set_threshold(
        self,
        val_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None,
        sigma_k: Optional[float] = None
    ) -> float:
        """
        Determine anomaly threshold from validation scores.

        The validation set should contain only normal sequences.
        The threshold is set such that most normal sequences fall below it.

        Methods:
            - "percentile": Use nth percentile of validation scores
            - "sigma": Use mean + k*std of validation scores

        Args:
            val_scores: Anomaly scores from normal validation sequences
            method: Threshold method (default from config)
            percentile: Percentile value for percentile method
            sigma_k: Number of std deviations for sigma method

        Returns:
            The computed threshold value
        """
        method = method or self.config.threshold_method
        percentile = percentile or self.config.threshold_percentile
        sigma_k = sigma_k or self.config.threshold_sigma

        if method == "percentile":
            self.threshold = np.percentile(val_scores, percentile)
            logger.info(f"Set threshold using {percentile}th percentile: {self.threshold:.4f}")
        elif method == "sigma":
            mean_score = np.mean(val_scores)
            std_score = np.std(val_scores)
            self.threshold = mean_score + sigma_k * std_score
            logger.info(
                f"Set threshold using μ + {sigma_k}σ: {self.threshold:.4f} "
                f"(μ={mean_score:.4f}, σ={std_score:.4f})"
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.threshold

    def set_point_threshold(
        self,
        val_point_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None,
        sigma_k: Optional[float] = None
    ) -> float:
        """
        Determine point-level anomaly threshold from validation scores.

        Args:
            val_point_scores: Point-level scores from normal validation, shape (N, T)
            method: "percentile" or "sigma" (default from config)
            percentile: Percentile for percentile method
            sigma_k: Multiplier for sigma method

        Returns:
            Point-level threshold τ
        """
        method = method or self.config.threshold_method
        percentile = percentile or self.config.threshold_percentile
        sigma_k = sigma_k or self.config.threshold_sigma

        # Flatten all point scores from normal validation windows
        all_scores = val_point_scores.flatten()

        if method == "percentile":
            self.point_threshold = float(np.percentile(all_scores, percentile))
            logger.info(f"Set point threshold using {percentile}th percentile: {self.point_threshold:.4f}")
        elif method == "sigma":
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            self.point_threshold = float(mean_score + sigma_k * std_score)
            logger.info(
                f"Set point threshold using μ + {sigma_k}σ: {self.point_threshold:.4f} "
                f"(μ={mean_score:.4f}, σ={std_score:.4f})"
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.point_threshold

    @staticmethod
    def find_optimal_threshold(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        method: str = "midpoint",
        beta: float = 1.0
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold using labeled normal and anomaly scores.

        This method enables threshold calibration using both normal validation
        data and synthetic (or real) anomaly examples, rather than relying
        solely on percentile-based thresholds from normal data.

        Methods:
            - "midpoint": Midpoint between max(normal) and min(anomaly)
              Falls back to 99th percentile of normal if distributions overlap.
            - "f1_max": Search for threshold that maximizes F1 (or F-beta) score
            - "youden": Maximize Youden's J statistic (sensitivity + specificity - 1)

        Args:
            normal_scores: Scores from normal validation windows
            anomaly_scores: Scores from synthetic/real anomaly windows
            method: Calibration method ("midpoint", "f1_max", "youden")
            beta: Beta for F-beta score (only used with f1_max method)

        Returns:
            Tuple of (optimal_threshold, calibration_metrics)
        """
        max_normal = np.max(normal_scores)
        min_anomaly = np.min(anomaly_scores)
        gap = min_anomaly - max_normal

        metrics = {
            "method": method,
            "max_normal": float(max_normal),
            "min_anomaly": float(min_anomaly),
            "gap": float(gap),
            "separable": gap > 0,
        }

        if method == "midpoint":
            if gap > 0:
                # Perfect separation possible - use midpoint
                threshold = (max_normal + min_anomaly) / 2
                metrics["threshold_source"] = "midpoint"
            else:
                # Overlap exists - fall back to high percentile of normal
                threshold = float(np.percentile(normal_scores, 99))
                metrics["threshold_source"] = "fallback_percentile_99"
                logger.warning(
                    f"Distributions overlap (gap={gap:.2f}), "
                    f"falling back to 99th percentile: {threshold:.4f}"
                )

            logger.info(f"Optimal threshold (midpoint): {threshold:.4f}")
            return threshold, metrics

        elif method == "f1_max":
            # Search over candidate thresholds
            all_scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate([
                np.zeros(len(normal_scores)),  # 0 = normal
                np.ones(len(anomaly_scores))   # 1 = anomaly
            ])

            # Generate candidate thresholds between min and max
            candidates = np.percentile(all_scores, np.arange(1, 100, 0.5))

            best_f_beta = 0
            best_threshold = float(np.median(all_scores))

            for candidate in candidates:
                predictions = all_scores > candidate
                tp = np.sum(predictions & labels.astype(bool))
                fp = np.sum(predictions & ~labels.astype(bool))
                fn = np.sum(~predictions & labels.astype(bool))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                if precision + recall > 0:
                    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
                else:
                    f_beta = 0

                if f_beta > best_f_beta:
                    best_f_beta = f_beta
                    best_threshold = float(candidate)

            metrics["best_f_beta"] = float(best_f_beta)
            metrics["beta"] = beta
            logger.info(f"Optimal threshold (F{beta:.1f} max): {best_threshold:.4f} (F={best_f_beta:.4f})")
            return best_threshold, metrics

        elif method == "youden":
            # Maximize Youden's J = sensitivity + specificity - 1
            all_scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate([
                np.zeros(len(normal_scores)),
                np.ones(len(anomaly_scores))
            ])

            candidates = np.percentile(all_scores, np.arange(1, 100, 0.5))

            best_j = -1
            best_threshold = float(np.median(all_scores))

            for candidate in candidates:
                predictions = all_scores > candidate
                tp = np.sum(predictions & labels.astype(bool))
                tn = np.sum(~predictions & ~labels.astype(bool))
                fp = np.sum(predictions & ~labels.astype(bool))
                fn = np.sum(~predictions & labels.astype(bool))

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                j = sensitivity + specificity - 1

                if j > best_j:
                    best_j = j
                    best_threshold = float(candidate)

            metrics["best_youden_j"] = float(best_j)
            logger.info(f"Optimal threshold (Youden): {best_threshold:.4f} (J={best_j:.4f})")
            return best_threshold, metrics

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on threshold.

        Args:
            scores: Anomaly scores, shape (num_sequences,)

        Returns:
            Boolean array where True = anomaly, shape (num_sequences,)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        return scores > self.threshold

    def predict_points(self, point_scores: np.ndarray) -> np.ndarray:
        """
        Predict point-level anomalies based on point threshold.

        Args:
            point_scores: Point-level anomaly scores, shape (num_sequences, seq_len)

        Returns:
            Boolean array where True = anomaly, shape (num_sequences, seq_len)
        """
        if self.point_threshold is None:
            raise ValueError("Point threshold not set. Call set_point_threshold() first.")

        return point_scores > self.point_threshold

    def predict_windows_from_points(
        self,
        point_predictions: np.ndarray,
        k: Optional[int] = None
    ) -> np.ndarray:
        """
        Derive window-level labels using HardCriterion.

        A window is flagged as anomalous if k or more points exceed the threshold.
        This implements the HardCriterion from Li's thesis.

        Args:
            point_predictions: Boolean array of point predictions, shape (num_sequences, seq_len)
            k: Number of anomalous points required to flag window (default from config)

        Returns:
            Boolean array of window predictions, shape (num_sequences,)
        """
        k = k or self.config.hard_criterion_k

        # Count anomalous points per window
        anomalous_point_counts = np.sum(point_predictions, axis=1)

        # Apply HardCriterion: window is anomalous if >= k points exceed threshold
        window_predictions = anomalous_point_counts >= k

        logger.debug(
            f"HardCriterion (k={k}): {window_predictions.sum()}/{len(window_predictions)} "
            f"windows flagged as anomalous"
        )

        return window_predictions

    def score_and_predict(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method to compute scores and predictions in one call.

        Args:
            model: Trained LSTM encoder-decoder model
            data_loader: DataLoader with sequences to evaluate
            device: Device to run inference on

        Returns:
            Dict with keys:
                - scores: Sequence-level anomaly scores (Mahalanobis distance)
                - errors: Raw reconstruction errors
                - predictions: Boolean anomaly predictions
        """
        scores, errors = self.compute_scores(model, data_loader, device)
        predictions = self.predict(scores)

        return {
            "scores": scores,
            "errors": errors,
            "predictions": predictions,
        }

    def get_stats(self) -> Dict:
        """Get scorer statistics."""
        stats = {
            "is_fitted": self.is_fitted,
            "threshold": self.threshold,
            "scoring_mode": self.config.scoring_mode,
            "config": {
                "threshold_method": self.config.threshold_method,
                "threshold_percentile": self.config.threshold_percentile,
                "threshold_sigma": self.config.threshold_sigma,
                "hard_criterion_k": self.config.hard_criterion_k,
            }
        }

        if self.mu is not None:
            stats["mu_range"] = (float(self.mu.min()), float(self.mu.max()))
            stats["cov_shape"] = self.cov.shape
            stats["cov_condition_number"] = float(np.linalg.cond(self.cov))

        # Point-level stats
        if self.mu_point is not None:
            stats["mu_point"] = float(self.mu_point[0]) if len(self.mu_point) == 1 else self.mu_point.tolist()
            stats["sigma_point"] = float(self.sigma_point[0]) if self.sigma_point is not None and len(self.sigma_point) == 1 else None
            stats["point_threshold"] = self.point_threshold

        return stats

    def save(self, filepath: str) -> None:
        """
        Save scorer state to file.

        Args:
            filepath: Path to save the scorer
        """
        state = {
            # Window-level attributes (legacy)
            "mu": self.mu,
            "cov": self.cov,
            "cov_inv": self.cov_inv,
            "threshold": self.threshold,
            "config": self.config,
            "is_fitted": self.is_fitted,
            # Point-level attributes (Malhotra et al. 2016)
            "mu_point": self.mu_point,
            "sigma_point": self.sigma_point,
            "sigma_inv_point": self.sigma_inv_point,
            "cov_point": self.cov_point,
            "cov_inv_point": self.cov_inv_point,
            "point_threshold": self.point_threshold,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved scorer to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AnomalyScorer":
        """
        Load scorer state from file.

        Handles both legacy (window-level only) and new (point-level) scorers.

        Args:
            filepath: Path to load the scorer from

        Returns:
            Loaded AnomalyScorer instance
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        scorer = cls(config=state["config"])
        # Window-level attributes (always present)
        scorer.mu = state["mu"]
        scorer.cov = state["cov"]
        scorer.cov_inv = state["cov_inv"]
        scorer.threshold = state["threshold"]
        scorer.is_fitted = state["is_fitted"]

        # Point-level attributes (may not be present in legacy scorers)
        scorer.mu_point = state.get("mu_point")
        scorer.sigma_point = state.get("sigma_point")
        scorer.sigma_inv_point = state.get("sigma_inv_point")
        scorer.cov_point = state.get("cov_point")
        scorer.cov_inv_point = state.get("cov_inv_point")
        scorer.point_threshold = state.get("point_threshold")

        logger.info(f"Loaded scorer from {filepath}")
        if scorer.mu_point is not None:
            logger.info(f"  Point-level scoring enabled (mu_point={scorer.mu_point[0]:.6f})")
        return scorer


def main():
    """Test the anomaly scorer with dummy data."""
    import sys
    sys.path.insert(0, ".")

    from lstm_autoencoder import create_model
    from data_preprocessor import NYCTaxiPreprocessor

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("ANOMALY SCORER TEST")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load and preprocess data
    print("\nLoading data...")
    preprocessor = NYCTaxiPreprocessor()
    dataloaders, _ = preprocessor.preprocess("data/nyc_taxi.csv", batch_size=4)

    # Create model (untrained - just for testing scorer mechanics)
    print("\nCreating model...")
    model = create_model(hidden_dim=64, num_layers=1)
    model.to(device)

    # Create scorer
    print("\nTesting AnomalyScorer...")
    scorer = AnomalyScorer()

    # Fit on training data
    print("\n1. Fitting error distribution on training data...")
    scorer.fit(model, dataloaders["train"], device)

    print(f"\nScorer stats after fit:")
    for key, value in scorer.get_stats().items():
        print(f"  {key}: {value}")

    # Compute scores on validation data (for threshold)
    print("\n2. Computing scores on validation data...")
    val_scores, val_errors = scorer.compute_scores(
        model, dataloaders["threshold_val"], device
    )
    print(f"  Validation scores shape: {val_scores.shape}")
    print(f"  Score range: [{val_scores.min():.4f}, {val_scores.max():.4f}]")

    # Set threshold
    print("\n3. Setting threshold...")
    threshold = scorer.set_threshold(val_scores, method="percentile", percentile=95)
    print(f"  Threshold (95th percentile): {threshold:.4f}")

    # Compute scores on test data
    print("\n4. Computing scores on test data...")
    test_scores, test_errors = scorer.compute_scores(
        model, dataloaders["test"], device
    )
    print(f"  Test scores shape: {test_scores.shape}")
    print(f"  Score range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")

    # Predict anomalies
    print("\n5. Predicting anomalies...")
    predictions = scorer.predict(test_scores)
    num_anomalies = predictions.sum()
    print(f"  Predicted anomalies: {num_anomalies}/{len(predictions)}")

    # Test save/load
    print("\n6. Testing save/load...")
    scorer.save("/tmp/test_scorer.pkl")
    loaded_scorer = AnomalyScorer.load("/tmp/test_scorer.pkl")
    print(f"  Loaded threshold: {loaded_scorer.threshold:.4f}")
    assert loaded_scorer.threshold == scorer.threshold, "Load failed!"

    # Get test week info for context
    print("\n7. Test week anomaly status:")
    test_week_info = preprocessor.get_test_week_info()
    for i, (score, pred, week) in enumerate(zip(test_scores, predictions, test_week_info)):
        status = "ANOMALY" if pred else "normal"
        actual = "HAS_ANOMALY" if week["is_anomaly"] else "normal"
        match = "✓" if (pred == week["is_anomaly"]) else "✗"
        print(f"  Week {week['year_week']}: score={score:.2f}, pred={status:7s}, actual={actual:10s} {match}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
