"""
Synthetic Anomaly Generator for Threshold Calibration

Provides functions to inject synthetic anomalies into normal time series
data for improved threshold calibration in anomaly detection.

This enables calibrating the anomaly threshold using both normal and
synthetic anomaly examples, rather than relying solely on percentile-based
thresholds from normal data.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyntheticAnomalyConfig:
    """Configuration for synthetic anomaly generation."""

    # Which anomaly types to generate
    anomaly_types: List[str] = field(
        default_factory=lambda: ["point", "level_shift", "temporal"]
    )

    # How many synthetic anomalies per normal week
    num_synthetic_per_normal: int = 1

    # Point anomaly parameters
    point_n_points: int = 5  # Number of points to corrupt
    point_magnitude: float = 3.0  # Magnitude in std units

    # Level shift parameters
    level_shift_fraction: float = 0.3  # Fraction of week to shift
    level_shift_magnitude: float = 2.0  # Magnitude in std units

    # Noise injection parameters
    noise_fraction: float = 0.3  # Fraction of week to add noise
    noise_scale: float = 0  # Noise magnitude in std units

    # Temporal shift parameters
    temporal_shift_hours: int = 6  # Hours to shift pattern

    # Reproducibility
    random_seed: Optional[int] = 42


class SyntheticAnomalyGenerator:
    """
    Generates synthetic anomalies from normal time series data.

    Supports multiple anomaly types that can be combined:
    - point: Random spikes/drops at individual timesteps
    - level_shift: Sustained shift over a portion of the sequence
    - noise: Gaussian noise injection over a portion
    - temporal: Circular shift of the pattern (breaks daily cycle)
    """

    def __init__(self, config: Optional[SyntheticAnomalyConfig] = None):
        self.config = config or SyntheticAnomalyConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def inject_point_anomalies(self, week: np.ndarray) -> np.ndarray:
        """
        Inject random spikes/drops at individual points.

        Args:
            week: Original week data, shape (seq_len,) or (seq_len, 1)

        Returns:
            Corrupted week with point anomalies
        """
        was_2d = week.ndim == 2
        if was_2d:
            week = week.squeeze(-1)

        corrupted = week.copy()
        seq_len = len(week)
        std = week.std()

        # Select random indices
        n_points = min(self.config.point_n_points, seq_len)
        indices = self._rng.choice(seq_len, size=n_points, replace=False)

        # Inject spikes/drops
        for idx in indices:
            sign = self._rng.choice([-1, 1])
            corrupted[idx] += sign * self.config.point_magnitude * std

        if was_2d:
            corrupted = corrupted.reshape(-1, 1)

        return corrupted

    def inject_level_shift(self, week: np.ndarray) -> np.ndarray:
        """
        Inject a sustained level shift over a portion of the week.

        Args:
            week: Original week data, shape (seq_len,) or (seq_len, 1)

        Returns:
            Corrupted week with level shift
        """
        was_2d = week.ndim == 2
        if was_2d:
            week = week.squeeze(-1)

        corrupted = week.copy()
        seq_len = len(week)
        std = week.std()

        # Determine shift region
        shift_len = int(seq_len * self.config.level_shift_fraction)
        max_start = seq_len - shift_len
        start = self._rng.integers(0, max_start + 1)

        # Apply shift
        sign = self._rng.choice([-1, 1])
        corrupted[start:start + shift_len] += (
            sign * self.config.level_shift_magnitude * std
        )

        if was_2d:
            corrupted = corrupted.reshape(-1, 1)

        return corrupted

    def inject_noise(self, week: np.ndarray) -> np.ndarray:
        """
        Inject Gaussian noise over a portion of the week.

        Args:
            week: Original week data, shape (seq_len,) or (seq_len, 1)

        Returns:
            Corrupted week with noise injection
        """
        was_2d = week.ndim == 2
        if was_2d:
            week = week.squeeze(-1)

        corrupted = week.copy()
        seq_len = len(week)
        std = week.std()

        # Determine noise region
        noise_len = int(seq_len * self.config.noise_fraction)
        max_start = seq_len - noise_len
        start = self._rng.integers(0, max_start + 1)

        # Generate and apply noise
        noise = self._rng.standard_normal(noise_len) * self.config.noise_scale * std
        corrupted[start:start + noise_len] += noise

        if was_2d:
            corrupted = corrupted.reshape(-1, 1)

        return corrupted

    def inject_temporal_shift(self, week: np.ndarray) -> np.ndarray:
        """
        Shift the time pattern by a few hours (breaks daily cycle).

        Args:
            week: Original week data, shape (seq_len,) or (seq_len, 1)

        Returns:
            Corrupted week with temporal shift
        """
        was_2d = week.ndim == 2
        if was_2d:
            week = week.squeeze(-1)

        # Shift by N hours (2 samples per hour for 30-min intervals)
        shift_samples = self.config.temporal_shift_hours * 2
        corrupted = np.roll(week, shift_samples)

        if was_2d:
            corrupted = corrupted.reshape(-1, 1)

        return corrupted

    def generate_synthetic_anomaly(
        self,
        week: np.ndarray,
        anomaly_type: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Generate a synthetic anomaly from a normal week.

        Args:
            week: Normal week data, shape (seq_len,) or (seq_len, 1)
            anomaly_type: Specific type or None for random selection

        Returns:
            Tuple of (corrupted_week, anomaly_type_used)
        """
        if anomaly_type is None:
            anomaly_type = self._rng.choice(self.config.anomaly_types)

        if anomaly_type == "point":
            corrupted = self.inject_point_anomalies(week)
        elif anomaly_type == "level_shift":
            corrupted = self.inject_level_shift(week)
        elif anomaly_type == "noise":
            corrupted = self.inject_noise(week)
        elif anomaly_type == "temporal":
            corrupted = self.inject_temporal_shift(week)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        return corrupted, anomaly_type

    def generate_synthetic_dataset(
        self,
        normal_weeks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate synthetic anomaly dataset from normal weeks.

        Args:
            normal_weeks: Array of normal weeks, shape (N, seq_len) or (N, seq_len, 1)

        Returns:
            Tuple of:
                - synthetic_weeks: Array of synthetic anomaly weeks
                - labels: Binary labels (1 = anomaly for all)
                - anomaly_types: List of anomaly types used for each
        """
        was_3d = normal_weeks.ndim == 3
        if was_3d:
            normal_weeks_2d = normal_weeks.squeeze(-1)
        else:
            normal_weeks_2d = normal_weeks

        n_normal = len(normal_weeks_2d)
        n_synthetic = n_normal * self.config.num_synthetic_per_normal

        synthetic_list = []
        types_list = []

        for i in range(n_synthetic):
            # Select a normal week to corrupt (cycle through)
            source_idx = i % n_normal
            source_week = normal_weeks_2d[source_idx]

            # Generate synthetic anomaly
            corrupted, anom_type = self.generate_synthetic_anomaly(source_week)
            synthetic_list.append(corrupted)
            types_list.append(anom_type)

        synthetic_weeks = np.array(synthetic_list)
        labels = np.ones(n_synthetic, dtype=np.int32)  # All are anomalies

        if was_3d:
            synthetic_weeks = synthetic_weeks.reshape(n_synthetic, -1, 1)

        logger.info(
            f"Generated {n_synthetic} synthetic anomalies from {n_normal} normal weeks"
        )
        logger.info(f"  Anomaly types used: {dict(zip(*np.unique(types_list, return_counts=True)))}")

        return synthetic_weeks, labels, types_list
