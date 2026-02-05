"""
LSTM Streaming Detector for Real-Time Anomaly Detection

Wraps the trained LSTM Encoder-Decoder model for use in the
Kafka/Spark streaming pipeline. Accumulates data into weekly
windows and runs detection when a complete window is available.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch

from base_detector import BaseDetector
from lstm_autoencoder import EncDecAD, ModelConfig
from anomaly_scorer import AnomalyScorer

logger = logging.getLogger(__name__)


class LSTMStreamingDetector(BaseDetector):
    """
    Streaming anomaly detector using pre-trained LSTM Encoder-Decoder.

    This detector:
    1. Loads a pre-trained LSTM model, scaler, and scorer
    2. Accumulates incoming data points into a buffer
    3. When enough data is available, runs anomaly detection
    4. Returns anomalies based on the trained threshold

    The model was trained on weekly windows (336 samples = 48/day * 7 days),
    but for streaming we use a sliding window approach with configurable
    step size for more responsive detection.

    Attributes:
        model: Trained EncDecAD model
        scaler: Fitted StandardScaler for normalization
        scorer: Fitted AnomalyScorer with threshold
        window_size: Number of samples per detection window
        buffer: Accumulated data points
    """

    def __init__(
        self,
        model_path: str = "models/lstm_model.pt",
        scaler_path: str = "models/scaler.pkl",
        scorer_path: str = "models/scorer.pkl",
        window_size: int = 336,
        min_samples: int = 336,
        device: Optional[str] = None,
    ):
        """
        Initialize the LSTM streaming detector.

        Args:
            model_path: Path to saved PyTorch model
            scaler_path: Path to saved StandardScaler
            scorer_path: Path to saved AnomalyScorer
            window_size: Samples per detection window (default: 336 = 1 week)
            min_samples: Minimum samples before detection runs
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scorer_path = scorer_path
        self.window_size = window_size
        self._min_samples = min_samples

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model components (loaded lazily)
        self.model: Optional[EncDecAD] = None
        self.scaler = None
        self.scorer: Optional[AnomalyScorer] = None

        # State
        self._is_loaded = False
        self._load_error: Optional[str] = None

        # Try to load model artifacts
        self._load_artifacts()

        logger.info(f"Initialized LSTMStreamingDetector")
        logger.info(f"  Window size: {self.window_size}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model loaded: {self._is_loaded}")

    def _load_artifacts(self) -> None:
        """Load model, scaler, and scorer from disk."""
        try:
            # Check if files exist
            model_path = Path(self.model_path)
            scaler_path = Path(self.scaler_path)
            scorer_path = Path(self.scorer_path)

            if not model_path.exists():
                self._load_error = f"Model file not found: {model_path}"
                logger.warning(self._load_error)
                return

            if not scaler_path.exists():
                self._load_error = f"Scaler file not found: {scaler_path}"
                logger.warning(self._load_error)
                return

            if not scorer_path.exists():
                self._load_error = f"Scorer file not found: {scorer_path}"
                logger.warning(self._load_error)
                return

            # Load model
            logger.info(f"Loading model from {model_path}")
            torch.serialization.add_safe_globals([ModelConfig])
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
            self.model = EncDecAD(config=checkpoint["model_config"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            # Load scaler
            logger.info(f"Loading scaler from {scaler_path}")
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            # Load scorer
            logger.info(f"Loading scorer from {scorer_path}")
            self.scorer = AnomalyScorer.load(str(scorer_path))

            self._is_loaded = True
            self._load_error = None

            logger.info("All model artifacts loaded successfully")
            logger.info(f"  Model config: {self.model.get_config()}")
            logger.info(f"  Scorer threshold: {self.scorer.threshold:.4f}")

        except Exception as e:
            self._load_error = f"Failed to load model artifacts: {e}"
            logger.error(self._load_error)
            self._is_loaded = False

    def _compute_sequence_score_and_errors(
        self,
        sequence: np.ndarray
    ) -> tuple[bool, float, np.ndarray, np.ndarray]:
        """
        Compute anomaly score and point-wise errors for a sequence.

        Supports both point-level (Malhotra paper) and window-level (legacy) scoring.

        Args:
            sequence: Normalized sequence of shape (seq_len,)

        Returns:
            Tuple of (is_anomaly, window_score, point_scores, point_predictions)
        """
        if self.model is None or self.scorer is None:
            return False, 0.0, np.zeros_like(sequence), np.zeros_like(sequence, dtype=bool)

        # Reshape for model: (1, seq_len, 1)
        x = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            x_reconstructed = self.model(x)

        # Compute point-wise error
        error = torch.abs(x - x_reconstructed).cpu().numpy().squeeze()

        # Check if point-level scoring is available
        use_point_level = (
            self.scorer.mu_point is not None and
            self.scorer.point_threshold is not None
        )

        if use_point_level:
            # Point-level scoring (Malhotra et al. 2016)
            # Compute per-point anomaly scores: (e - μ)² / σ²
            point_scores = ((error - self.scorer.mu_point[0]) ** 2) / self.scorer.sigma_point[0]

            # Point-level predictions
            point_predictions = point_scores > self.scorer.point_threshold

            # Window-level decision using HardCriterion
            k = self.scorer.config.hard_criterion_k
            num_anomalous_points = np.sum(point_predictions)
            is_anomaly = num_anomalous_points >= k

            # Window score: max point score for reporting
            window_score = np.max(point_scores)

            return is_anomaly, window_score, point_scores, point_predictions
        else:
            # Legacy window-level scoring
            mahalanobis_score = self.scorer._mahalanobis_distance(error)
            is_anomaly = mahalanobis_score > self.scorer.threshold

            # Compute legacy point-wise scores for visualization
            point_scores = ((error - self.scorer.mu) ** 2) / (
                np.diag(self.scorer.cov) + 1e-8
            )

            return is_anomaly, mahalanobis_score, point_scores, np.zeros_like(error, dtype=bool)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the provided data.

        For LSTM detector, we analyze complete windows of data.
        Supports both point-level (Malhotra paper) and window-level (legacy) scoring.

        Point-level mode: Uses HardCriterion (k points > τ) for window decision,
        returns the specific points that exceeded the point threshold.

        Window-level mode: Uses Mahalanobis distance for window decision,
        returns top 5% highest reconstruction error points.

        Args:
            df: DataFrame with 'timestamp' and 'value' columns

        Returns:
            DataFrame with anomalous records and anomaly scores
        """
        if not self._is_loaded:
            logger.warning(f"Model not loaded: {self._load_error}")
            return pd.DataFrame()

        if len(df) < self._min_samples:
            logger.debug(
                f"Not enough samples for LSTM detection: "
                f"{len(df)} < {self._min_samples}"
            )
            return pd.DataFrame()

        # Extract values and normalize
        values = df["value"].values.astype(float).reshape(-1, 1)
        values_normalized = self.scaler.transform(values).flatten()

        # Analyze the most recent complete window
        window_data = values_normalized[-self.window_size:]
        is_anomaly, window_score, point_scores, point_predictions = self._compute_sequence_score_and_errors(
            window_data
        )

        # Check if point-level scoring is being used
        use_point_level = (
            self.scorer.mu_point is not None and
            self.scorer.point_threshold is not None
        )

        # Log detailed scoring information
        logger.info(
            f"  Scoring: raw_range=[{values.min():.0f}, {values.max():.0f}], "
            f"normalized_range=[{values_normalized.min():.4f}, {values_normalized.max():.4f}]"
        )

        if use_point_level:
            num_anomalous = point_predictions.sum()
            k = self.scorer.config.hard_criterion_k
            logger.info(
                f"  Point-level: {num_anomalous} pts > τ={self.scorer.point_threshold:.2f} "
                f"(k={k}) -> {'ANOMALY' if is_anomaly else 'NORMAL'}"
            )
        else:
            logger.info(
                f"  Window-level: score={window_score:.2f} vs threshold={self.scorer.threshold:.2f} "
                f"-> {'ANOMALY' if is_anomaly else 'NORMAL'}"
            )

        if is_anomaly:
            # Get the window DataFrame
            window_df = df.tail(self.window_size).copy().reset_index(drop=True)
            window_df["point_score"] = point_scores

            if use_point_level:
                # Return points that exceeded the point threshold
                anomalous_mask = point_predictions
                anomalous_points = window_df[anomalous_mask].copy()
                anomalous_points["anomaly_score"] = window_score

                logger.info(
                    f"LSTM detected anomaly (point-level)! "
                    f"{point_predictions.sum()} points exceed τ={self.scorer.point_threshold:.2f}, "
                    f"returning {len(anomalous_points)} anomalous points"
                )
            else:
                # Legacy: use top 5% of point scores
                point_threshold = np.percentile(point_scores, 95)
                anomalous_mask = point_scores >= point_threshold
                anomalous_points = window_df[anomalous_mask].copy()
                anomalous_points["anomaly_score"] = window_score

                logger.info(
                    f"LSTM detected anomaly (window-level)! "
                    f"Score: {window_score:.2f} (threshold: {self.scorer.threshold:.2f}), "
                    f"returning {len(anomalous_points)} high-error points"
                )

            return anomalous_points[["timestamp", "value", "anomaly_score"]]

        return pd.DataFrame()

    def detect_window(
        self,
        timestamps: List[str],
        values: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect anomaly in a specific window of data.

        Alternative interface for when data comes as separate lists.

        Args:
            timestamps: List of timestamp strings
            values: List of values

        Returns:
            Dict with anomaly info if detected, None otherwise
        """
        if not self._is_loaded:
            return None

        if len(values) < self.window_size:
            return None

        # Normalize
        values_array = np.array(values[-self.window_size:]).reshape(-1, 1)
        values_normalized = self.scaler.transform(values_array).flatten()

        # Compute score
        is_anomaly, window_score, _, point_predictions = self._compute_sequence_score_and_errors(
            values_normalized
        )

        if is_anomaly:
            result = {
                "is_anomaly": True,
                "score": float(window_score),
                "start_time": timestamps[-self.window_size],
                "end_time": timestamps[-1],
                "window_size": self.window_size,
            }

            # Add threshold info based on scoring mode
            use_point_level = self.scorer.mu_point is not None and self.scorer.point_threshold is not None
            if use_point_level:
                result["threshold"] = float(self.scorer.point_threshold)
                result["anomalous_points"] = int(point_predictions.sum())
                result["hard_criterion_k"] = self.scorer.config.hard_criterion_k
            else:
                result["threshold"] = float(self.scorer.threshold)

            return result

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        stats = {
            "detector_type": "lstm",
            "is_loaded": self._is_loaded,
            "window_size": self.window_size,
            "min_samples": self._min_samples,
            "device": str(self.device),
        }

        if self._is_loaded and self.scorer is not None:
            stats["threshold"] = float(self.scorer.threshold) if self.scorer.threshold else None
            stats["threshold_method"] = self.scorer.config.threshold_method
            stats["scoring_mode"] = self.scorer.config.scoring_mode

            # Point-level scoring info
            if self.scorer.mu_point is not None:
                stats["point_threshold"] = float(self.scorer.point_threshold) if self.scorer.point_threshold else None
                stats["hard_criterion_k"] = self.scorer.config.hard_criterion_k
                stats["mu_point"] = float(self.scorer.mu_point[0])
                stats["sigma_point"] = float(self.scorer.sigma_point[0]) if self.scorer.sigma_point is not None else None

        if self._load_error:
            stats["load_error"] = self._load_error

        if self._is_loaded and self.model is not None:
            stats["model_config"] = self.model.get_config()

        return stats

    def get_name(self) -> str:
        """Get the detector name for display purposes."""
        return "LSTM Encoder-Decoder"

    @property
    def is_ready(self) -> bool:
        """Check if the detector is ready to perform detection."""
        return self._is_loaded

    @property
    def min_samples_required(self) -> int:
        """Minimum number of samples required for detection."""
        return self._min_samples
