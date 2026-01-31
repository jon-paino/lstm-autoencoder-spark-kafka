"""
Isolation Forest Anomaly Detector

Sliding window-based anomaly detection using Isolation Forest algorithm.
Designed for real-time streaming data with configurable window size and
detection parameters.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Configuration for the Isolation Forest detector."""
    window_size: int = 200
    min_samples: int = 50
    contamination: float = 0.05
    n_estimators: int = 100
    random_state: int = 42
    feature_columns: list = field(default_factory=lambda: ["value"])


class IsolationForestDetector:
    """
    Sliding window anomaly detector using Isolation Forest.

    The detector maintains a sliding window of recent data points and
    periodically fits an Isolation Forest model to detect anomalies.

    Attributes:
        config: DetectorConfig with detection parameters
        model: Fitted IsolationForest model
        scaler: StandardScaler for feature normalization
        is_fitted: Whether the model has been fitted
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False

        logger.info(f"Initialized IsolationForestDetector with config: {self.config}")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from raw data.

        Creates features from the value column:
        - Raw value
        - Rolling mean (window=10)
        - Rolling std (window=10)
        - Deviation from rolling mean

        Args:
            df: DataFrame with at least 'value' column

        Returns:
            Feature matrix as numpy array
        """
        features_df = pd.DataFrame()

        # Primary feature: the value itself
        features_df["value"] = df["value"].astype(float)

        # Rolling statistics for context
        features_df["rolling_mean"] = (
            features_df["value"]
            .rolling(window=10, min_periods=1)
            .mean()
        )
        features_df["rolling_std"] = (
            features_df["value"]
            .rolling(window=10, min_periods=1)
            .std()
            .fillna(0)
        )

        # Deviation from expected value
        features_df["deviation"] = abs(
            features_df["value"] - features_df["rolling_mean"]
        )

        # Rate of change
        features_df["diff"] = features_df["value"].diff().fillna(0)

        return features_df.values

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the Isolation Forest model on the provided data.

        Args:
            df: DataFrame with 'value' column
        """
        if len(df) < self.config.min_samples:
            logger.warning(
                f"Not enough samples to fit: {len(df)} < {self.config.min_samples}"
            )
            return

        features = self._prepare_features(df)

        # Normalize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1  # Use all available cores
        )
        self.model.fit(features_scaled)
        self.is_fitted = True

        logger.debug(f"Fitted Isolation Forest on {len(df)} samples")

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the provided data.

        Fits the model if not already fitted, then predicts anomalies.
        Returns a DataFrame containing only the anomalous records.

        Args:
            df: DataFrame with 'value' and 'timestamp' columns

        Returns:
            DataFrame with anomalous records and anomaly scores
        """
        if len(df) < self.config.min_samples:
            logger.debug(
                f"Not enough samples for detection: {len(df)} < {self.config.min_samples}"
            )
            return pd.DataFrame()

        # Fit on current window
        self.fit(df)

        if not self.is_fitted or self.model is None or self.scaler is None:
            return pd.DataFrame()

        # Prepare and scale features
        features = self._prepare_features(df)
        features_scaled = self.scaler.transform(features)

        # Predict: -1 for anomalies, 1 for normal
        predictions = self.model.predict(features_scaled)

        # Get anomaly scores (lower = more anomalous)
        scores = self.model.score_samples(features_scaled)

        # Create result DataFrame
        result_df = df.copy()
        result_df["is_anomaly"] = predictions == -1
        result_df["anomaly_score"] = scores

        # Filter to anomalies only
        anomalies = result_df[result_df["is_anomaly"]].copy()

        if len(anomalies) > 0:
            logger.info(
                f"Detected {len(anomalies)} anomalies out of {len(df)} samples "
                f"({100 * len(anomalies) / len(df):.1f}%)"
            )

        return anomalies.drop(columns=["is_anomaly"])

    def detect_single(self, df: pd.DataFrame, new_point: dict) -> bool:
        """
        Check if a single new data point is anomalous.

        Uses the currently fitted model to classify a new point.

        Args:
            df: Historical data for context (for feature calculation)
            new_point: Dict with 'value' key

        Returns:
            True if the point is anomalous, False otherwise
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            return False

        # Append new point to calculate features with context
        temp_df = pd.concat([
            df.tail(20),  # Use last 20 points for rolling calculations
            pd.DataFrame([new_point])
        ], ignore_index=True)

        features = self._prepare_features(temp_df)
        last_features = features[-1:, :]  # Get features for new point only

        features_scaled = self.scaler.transform(last_features)
        prediction = self.model.predict(features_scaled)

        return prediction[0] == -1

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "is_fitted": self.is_fitted,
            "window_size": self.config.window_size,
            "min_samples": self.config.min_samples,
            "contamination": self.config.contamination,
            "n_estimators": self.config.n_estimators,
        }
