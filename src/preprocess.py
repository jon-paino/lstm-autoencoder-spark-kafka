"""
Data Preprocessing for LSTM Encoder-Decoder Anomaly Detection

Handles loading, parsing, segmentation, normalization, and splitting
of NYC taxi demand data for the LSTM autoencoder model.

Based on Malhotra et al. (2016) EncDec-AD approach:
- Segments data into weekly chunks (336 records = 48/day x 7 days)
- Fits scaler on training data only
- Creates train/val/test splits with normal data for training
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Constants for NYC taxi data
SAMPLES_PER_DAY = 48  # 30-minute intervals
SAMPLES_PER_WEEK = SAMPLES_PER_DAY * 7  # 336 samples per week

# Known anomaly windows in NYC taxi dataset
ANOMALY_WINDOWS = [
    ("2014-11-01 19:00:00", "2014-11-03 22:30:00"),  # NYC Marathon
    ("2014-11-25 12:00:00", "2014-11-29 19:00:00"),  # Thanksgiving
    ("2014-12-23 11:30:00", "2014-12-27 18:30:00"),  # Christmas
    ("2014-12-29 21:30:00", "2015-01-03 04:30:00"),  # New Year's
    ("2015-01-24 20:30:00", "2015-01-29 03:30:00"),  # Blizzard
]

EDGE_CASE_WEEKS = {
    "2015-01-04",
    "2015-01-18",
}


@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessing."""
    sequence_length: int = SAMPLES_PER_WEEK  # 336 (one week)
    train_weeks: int = 9
    val_weeks: int = 3     # For early stopping AND error distribution fitting
    threshold_weeks: int = 2  # For threshold calibration
    offset_days: int = 5   # Days to offset from data start (Sunday-Saturday alignment)
    min_anomaly_overlap: float = 0.10  # Minimum overlap fraction to label week anomalous


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for weekly time series sequences."""

    def __init__(self, sequences: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        if self.sequences.ndim == 2:
            self.sequences = self.sequences.unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


# ---------------------------------------------------------------------------
# Preprocessing functions (replacing NYCTaxiPreprocessor class)
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and parse the NYC taxi CSV file.

    Returns:
        DataFrame with parsed timestamp and value columns
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} records from {filepath}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Value range: {df['value'].min()} to {df['value'].max()}")

    return df


def _week_contains_anomaly(
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    min_overlap: float = 0.10,
) -> bool:
    """Check if a week contains sufficient anomaly overlap to be labeled anomalous."""
    for start, end in ANOMALY_WINDOWS:
        win_start = pd.Timestamp(start)
        win_end = pd.Timestamp(end)

        if week_start <= win_end and week_end >= win_start:
            overlap_start = max(win_start, week_start)
            overlap_end = min(win_end, week_end)
            overlap_intervals = int((overlap_end - overlap_start).total_seconds() / 1800) + 1
            overlap_fraction = overlap_intervals / SAMPLES_PER_WEEK

            if overlap_fraction >= min_overlap:
                return True

    return False


def segment_into_weeks(
    df: pd.DataFrame,
    config: Optional[PreprocessorConfig] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Segment the time series into weekly chunks using offset-based bracketing.

    Weeks are fixed 7-day windows (336 samples) starting from data_start + offset_days.

    Returns:
        Tuple of:
            - Array of shape (num_weeks, 336) containing values
            - List of dicts with week metadata (start_date, end_date, is_anomaly)
    """
    config = config or PreprocessorConfig()
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    data_start = df["timestamp"].min()
    offset_start = data_start + pd.Timedelta(days=config.offset_days)

    df_subset = df[df["timestamp"] >= offset_start].reset_index(drop=True)
    num_complete_weeks = len(df_subset) // SAMPLES_PER_WEEK

    weeks = []
    week_info = []

    for w in range(num_complete_weeks):
        start_idx = w * SAMPLES_PER_WEEK
        end_idx = start_idx + SAMPLES_PER_WEEK

        week_data = df_subset.iloc[start_idx:end_idx]
        values = week_data["value"].values.astype(np.float32)

        week_start = week_data["timestamp"].iloc[0]
        week_end = week_data["timestamp"].iloc[-1]
        week_name = week_start.strftime("%Y-%m-%d")

        contains_anomaly = _week_contains_anomaly(
            week_start, week_end, config.min_anomaly_overlap
        )

        weeks.append(values)
        week_info.append({
            "year_week": week_name,
            "start_date": week_start,
            "end_date": week_end,
            "is_anomaly": contains_anomaly,
            "is_edge_case": week_name in EDGE_CASE_WEEKS,
            "mean_value": values.mean(),
            "std_value": values.std(),
        })

    skipped_records = len(df) - len(df_subset)
    if skipped_records > 0:
        logger.debug(f"Skipped {skipped_records} records before offset start")

    tail_records = len(df_subset) % SAMPLES_PER_WEEK
    if tail_records > 0:
        logger.debug(f"Skipped {tail_records} records at end (incomplete week)")

    weekly_data = np.array(weeks)

    num_normal = sum(1 for w in week_info if not w["is_anomaly"])
    num_anomaly = sum(1 for w in week_info if w["is_anomaly"])

    logger.info(f"Segmented into {len(weeks)} complete weeks (offset={config.offset_days} days)")
    logger.info(f"  Week start day: {offset_start.strftime('%A')}")
    logger.info(f"  Normal weeks: {num_normal}")
    logger.info(f"  Weeks with anomalies: {num_anomaly}")

    return weekly_data, week_info


def create_splits(
    weekly_data: np.ndarray,
    week_info: List[Dict],
    config: Optional[PreprocessorConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
    """
    Create train/validation/test splits following the paper's methodology.

    Returns:
        Tuple of:
            - Dict with keys: train, val, threshold_val, test (arrays)
            - Dict with keys: train, val, threshold_val, test (index lists)
    """
    config = config or PreprocessorConfig()

    normal_indices = [i for i, w in enumerate(week_info) if not w["is_anomaly"]]

    train_end = config.train_weeks
    train_indices = normal_indices[:train_end]

    val_end = train_end + config.val_weeks
    val_indices = normal_indices[train_end:val_end]

    threshold_end = val_end + config.threshold_weeks
    threshold_indices = normal_indices[val_end:threshold_end]

    all_used = set(train_indices + val_indices + threshold_indices)
    test_indices = [i for i in range(len(weekly_data)) if i not in all_used]

    splits = {
        "train": weekly_data[train_indices] if train_indices else np.array([]),
        "val": weekly_data[val_indices] if val_indices else np.array([]),
        "threshold_val": weekly_data[threshold_indices] if threshold_indices else np.array([]),
        "test": weekly_data[test_indices] if test_indices else np.array([]),
    }

    split_indices = {
        "train": train_indices,
        "val": val_indices,
        "threshold_val": threshold_indices,
        "test": test_indices,
    }

    logger.info("Created data splits:")
    logger.info(f"  Train: {len(train_indices)} weeks (indices {train_indices})")
    logger.info(f"  Val (early stopping): {len(val_indices)} weeks")
    logger.info(f"  Val (threshold): {len(threshold_indices)} weeks")
    logger.info(f"  Test: {len(test_indices)} weeks")

    test_anomaly_weeks = [
        week_info[i]["year_week"]
        for i in test_indices
        if week_info[i]["is_anomaly"]
    ]
    logger.info(f"    Test anomaly weeks: {test_anomaly_weeks}")

    return splits, split_indices


def normalize(
    splits: Dict[str, np.ndarray],
    fit_on: str = "train",
) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    """
    Normalize data using StandardScaler (fit on training data only).

    Returns:
        Tuple of (normalized_splits, fitted_scaler)
    """
    if fit_on not in splits or len(splits[fit_on]) == 0:
        raise ValueError(f"Cannot fit scaler: '{fit_on}' split is empty")

    train_flat = splits[fit_on].flatten().reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(train_flat)

    logger.info(f"Fitted scaler on {fit_on} data:")
    logger.info(f"  Mean: {scaler.mean_[0]:.2f}")
    logger.info(f"  Std: {scaler.scale_[0]:.2f}")

    normalized_splits = {}
    for name, data in splits.items():
        if len(data) == 0:
            normalized_splits[name] = data
            continue
        original_shape = data.shape
        flat = data.flatten().reshape(-1, 1)
        normalized = scaler.transform(flat)
        normalized_splits[name] = normalized.reshape(original_shape)

    return normalized_splits, scaler


def create_dataloaders(
    normalized_splits: Dict[str, np.ndarray],
    batch_size: int = 4,
) -> Dict[str, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for each split.

    Data is NOT shuffled to preserve temporal ordering.
    """
    dataloaders = {}

    for name, data in normalized_splits.items():
        if len(data) == 0:
            dataloaders[name] = None
            continue

        dataset = TimeSeriesDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(data)),
            shuffle=False,
            drop_last=False,
        )
        dataloaders[name] = loader
        logger.debug(f"Created DataLoader for {name}: {len(dataset)} samples")

    return dataloaders


def get_test_week_info(
    week_info: List[Dict],
    split_indices: Dict[str, List[int]],
) -> List[Dict]:
    """Get metadata for test weeks."""
    return [week_info[i] for i in split_indices.get("test", [])]


def get_test_timestamps(
    week_info: List[Dict],
    split_indices: Dict[str, List[int]],
    sequence_length: int = SAMPLES_PER_WEEK,
) -> List[List[str]]:
    """
    Get timestamps for each test week (useful for localization).

    Returns:
        List of timestamp lists, one per test week.
    """
    test_timestamps = []
    for i in split_indices.get("test", []):
        week = week_info[i]
        start_date = week["start_date"]
        timestamps = [
            (start_date + pd.Timedelta(minutes=30 * j)).isoformat()
            for j in range(sequence_length)
        ]
        test_timestamps.append(timestamps)
    return test_timestamps


def inverse_transform(scaler: StandardScaler, data: np.ndarray) -> np.ndarray:
    """Inverse transform normalized data back to original scale."""
    original_shape = data.shape
    flat = data.flatten().reshape(-1, 1)
    original = scaler.inverse_transform(flat)
    return original.reshape(original_shape)


def preprocess_pipeline(
    filepath: str,
    config: Optional[PreprocessorConfig] = None,
    batch_size: int = 4,
) -> Tuple[Dict[str, Optional[DataLoader]], Dict[str, np.ndarray], StandardScaler, List[Dict], Dict[str, List[int]]]:
    """
    Run the complete preprocessing pipeline.

    Returns:
        Tuple of (dataloaders, normalized_splits, scaler, week_info, split_indices)
    """
    config = config or PreprocessorConfig()

    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)

    df = load_data(filepath)
    weekly_data, week_info = segment_into_weeks(df, config)
    splits, split_indices = create_splits(weekly_data, week_info, config)
    normalized_splits, scaler = normalize(splits)
    dataloaders = create_dataloaders(normalized_splits, batch_size=batch_size)

    logger.info("=" * 60)
    logger.info("Preprocessing complete")
    logger.info("=" * 60)

    return dataloaders, normalized_splits, scaler, week_info, split_indices
