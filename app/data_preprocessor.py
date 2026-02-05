"""
Data Preprocessor for LSTM Encoder-Decoder Anomaly Detection

Handles loading, parsing, segmentation, normalization, and splitting
of NYC taxi demand data for the LSTM autoencoder model.

Based on Malhotra et al. (2016) EncDec-AD approach:
- Segments data into weekly chunks (336 records = 48/day × 7 days)
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

# Known anomaly windows in NYC taxi dataset (precise timestamps from dataset labels)
ANOMALY_WINDOWS = [
    ("2014-11-01 19:00:00", "2014-11-03 22:30:00"),  # NYC Marathon (point: Nov 1 19:00)
    ("2014-11-25 12:00:00", "2014-11-29 19:00:00"),  # Thanksgiving
    ("2014-12-23 11:30:00", "2014-12-27 18:30:00"),  # Christmas
    ("2014-12-29 21:30:00", "2015-01-03 04:30:00"),  # New Year's
    ("2015-01-24 20:30:00", "2015-01-29 03:30:00"),  # Blizzard
]


@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessing."""
    sequence_length: int = SAMPLES_PER_WEEK  # 336 (one week)
    train_weeks: int = 9   # Weeks for training
    val_weeks: int = 3     # Weeks for early stopping AND error distribution fitting
    threshold_weeks: int = 2  # Weeks for threshold calibration
    # Remaining weeks for testing

    # Week bracketing configuration
    offset_days: int = 5   # Days to offset from data start (5 = Sunday-Saturday weeks)
                           # This ensures all anomaly points (including Jan 27 blizzard)
                           # fall within complete weeks in the test set
    min_anomaly_overlap: float = 0.10  # Minimum fraction of week that must overlap with
                                        # anomaly window to label week as anomalous (10%)
                                        # This prevents labeling weeks with tiny overlap


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for weekly time series sequences.

    Each sample is a complete week of taxi demand data,
    shaped as (sequence_length, num_features) for LSTM input.
    """

    def __init__(self, sequences: np.ndarray):
        """
        Args:
            sequences: Array of shape (num_weeks, sequence_length)
        """
        self.sequences = torch.FloatTensor(sequences)
        # Add feature dimension: (num_weeks, seq_len) -> (num_weeks, seq_len, 1)
        if self.sequences.ndim == 2:
            self.sequences = self.sequences.unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


class NYCTaxiPreprocessor:
    """
    Preprocessor for NYC taxi demand data.

    Handles the complete preprocessing pipeline:
    1. Load and parse CSV data
    2. Segment into weekly chunks
    3. Identify normal vs anomalous weeks
    4. Create train/val/test splits
    5. Normalize using StandardScaler (fit on train only)
    6. Generate PyTorch DataLoaders
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        self.scaler: Optional[StandardScaler] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.weekly_data: Optional[np.ndarray] = None
        self.week_info: Optional[List[Dict]] = None

        logger.info(f"Initialized NYCTaxiPreprocessor with config: {self.config}")

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and parse the NYC taxi CSV file.

        Args:
            filepath: Path to nyc_taxi.csv

        Returns:
            DataFrame with parsed timestamp and value columns
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        self.raw_df = df

        logger.info(f"Loaded {len(df)} records from {filepath}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Value range: {df['value'].min()} to {df['value'].max()}")

        return df

    def _is_in_anomaly_window(self, timestamp: pd.Timestamp) -> bool:
        """Check if a timestamp falls within any known anomaly window."""
        for start, end in ANOMALY_WINDOWS:
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            if start_dt <= timestamp <= end_dt:
                return True
        return False

    def _week_contains_anomaly(self, week_start: pd.Timestamp, week_end: pd.Timestamp) -> bool:
        """
        Check if a week contains sufficient anomaly overlap to be labeled anomalous.

        A week is labeled anomalous if the overlap with any anomaly window
        exceeds the configured min_anomaly_overlap threshold (default 10%).

        Args:
            week_start: Start timestamp of the week
            week_end: End timestamp of the week

        Returns:
            True if week should be labeled as anomalous
        """
        for start, end in ANOMALY_WINDOWS:
            win_start = pd.Timestamp(start)
            win_end = pd.Timestamp(end)

            # Check for overlap
            if week_start <= win_end and week_end >= win_start:
                # Calculate overlap
                overlap_start = max(win_start, week_start)
                overlap_end = min(win_end, week_end)

                # Count overlap in 30-minute intervals
                overlap_intervals = int((overlap_end - overlap_start).total_seconds() / 1800) + 1
                overlap_fraction = overlap_intervals / SAMPLES_PER_WEEK

                if overlap_fraction >= self.config.min_anomaly_overlap:
                    return True

        return False

    def segment_into_weeks(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Segment the time series into weekly chunks using offset-based bracketing.

        Weeks are fixed 7-day windows (336 samples) starting from data_start + offset_days.
        This approach ensures all known anomaly points fall within complete weeks,
        particularly the Jan 27, 2015 blizzard which would otherwise be in an
        incomplete week with ISO calendar alignment.

        Args:
            df: DataFrame with timestamp and value columns

        Returns:
            Tuple of:
                - Array of shape (num_weeks, 336) containing values
                - List of dicts with week metadata (start_date, end_date, is_anomaly)
        """
        if df is None:
            df = self.raw_df
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = df.copy().sort_values("timestamp").reset_index(drop=True)

        # Apply offset to start date
        data_start = df["timestamp"].min()
        offset_start = data_start + pd.Timedelta(days=self.config.offset_days)

        # Filter to data after offset
        df_subset = df[df["timestamp"] >= offset_start].reset_index(drop=True)

        # Calculate number of complete weeks
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

            # Use date-based naming (e.g., "2015-01-25" for start date)
            week_name = week_start.strftime("%Y-%m-%d")

            contains_anomaly = self._week_contains_anomaly(week_start, week_end)

            weeks.append(values)
            week_info.append({
                "year_week": week_name,
                "start_date": week_start,
                "end_date": week_end,
                "is_anomaly": contains_anomaly,
                "mean_value": values.mean(),
                "std_value": values.std(),
            })

        # Log skipped data due to offset
        skipped_records = len(df) - len(df_subset)
        if skipped_records > 0:
            logger.debug(
                f"Skipped {skipped_records} records before offset start "
                f"({data_start} to {offset_start})"
            )

        # Log incomplete tail
        tail_records = len(df_subset) % SAMPLES_PER_WEEK
        if tail_records > 0:
            logger.debug(
                f"Skipped {tail_records} records at end (incomplete week)"
            )

        self.weekly_data = np.array(weeks)
        self.week_info = week_info

        num_normal = sum(1 for w in week_info if not w["is_anomaly"])
        num_anomaly = sum(1 for w in week_info if w["is_anomaly"])

        logger.info(f"Segmented into {len(weeks)} complete weeks (offset={self.config.offset_days} days)")
        logger.info(f"  Week start day: {offset_start.strftime('%A')}")
        logger.info(f"  Normal weeks: {num_normal}")
        logger.info(f"  Weeks with anomalies: {num_anomaly}")

        return self.weekly_data, self.week_info

    def create_splits(
        self,
        weekly_data: Optional[np.ndarray] = None,
        week_info: Optional[List[Dict]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create train/validation/test splits following the paper's methodology.

        Split strategy (based on Malhotra et al.):
        - sN (train): First N normal weeks for training
        - vN1 (val): Next M normal weeks for early stopping + error distribution
        - vN2 (threshold_val): Additional normal weeks for threshold calibration
        - Test: Remaining weeks (normal + anomalies)

        Args:
            weekly_data: Array of shape (num_weeks, 336)
            week_info: List of week metadata dicts

        Returns:
            Dict with keys: train, val, threshold_val, test
            Each value is an array of shape (num_weeks, 336)
        """
        if weekly_data is None:
            weekly_data = self.weekly_data
        if week_info is None:
            week_info = self.week_info

        if weekly_data is None or week_info is None:
            raise ValueError("No segmented data. Call segment_into_weeks() first.")

        # Identify normal weeks
        normal_indices = [i for i, w in enumerate(week_info) if not w["is_anomaly"]]

        # Training set: first N normal weeks
        train_end = self.config.train_weeks
        train_indices = normal_indices[:train_end]

        # Validation set for early stopping: next M normal weeks
        val_end = train_end + self.config.val_weeks
        val_indices = normal_indices[train_end:val_end]

        # Threshold calibration set (vN2): additional normal weeks
        threshold_end = val_end + self.config.threshold_weeks
        threshold_indices = normal_indices[val_end:threshold_end]

        # Test set: remaining normal weeks + all anomaly weeks
        all_used = set(train_indices + val_indices + threshold_indices)
        test_indices = [i for i in range(len(weekly_data)) if i not in all_used]

        splits = {
            "train": weekly_data[train_indices] if train_indices else np.array([]),
            "val": weekly_data[val_indices] if val_indices else np.array([]),
            "threshold_val": weekly_data[threshold_indices] if threshold_indices else np.array([]),
            "test": weekly_data[test_indices] if test_indices else np.array([]),
        }

        # Store indices for later reference
        self.split_indices = {
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

        # Log which weeks contain anomalies in test set
        test_anomaly_weeks = [
            week_info[i]["year_week"]
            for i in test_indices
            if week_info[i]["is_anomaly"]
        ]
        logger.info(f"    Test anomaly weeks: {test_anomaly_weeks}")

        return splits

    def normalize(
        self,
        splits: Dict[str, np.ndarray],
        fit_on: str = "train"
    ) -> Dict[str, np.ndarray]:
        """
        Normalize data using StandardScaler.

        IMPORTANT: Scaler is fit only on training data to prevent data leakage.

        Args:
            splits: Dict of data splits from create_splits()
            fit_on: Which split to fit the scaler on (default: "train")

        Returns:
            Dict with same keys, but normalized values
        """
        if fit_on not in splits or len(splits[fit_on]) == 0:
            raise ValueError(f"Cannot fit scaler: '{fit_on}' split is empty")

        # Fit scaler on training data
        # Reshape to (n_samples, 1) for StandardScaler
        train_flat = splits[fit_on].flatten().reshape(-1, 1)

        self.scaler = StandardScaler()
        self.scaler.fit(train_flat)

        logger.info(f"Fitted scaler on {fit_on} data:")
        logger.info(f"  Mean: {self.scaler.mean_[0]:.2f}")
        logger.info(f"  Std: {self.scaler.scale_[0]:.2f}")

        # Transform all splits
        normalized_splits = {}
        for name, data in splits.items():
            if len(data) == 0:
                normalized_splits[name] = data
                continue

            original_shape = data.shape
            flat = data.flatten().reshape(-1, 1)
            normalized = self.scaler.transform(flat)
            normalized_splits[name] = normalized.reshape(original_shape)

        return normalized_splits

    def create_dataloaders(
        self,
        normalized_splits: Dict[str, np.ndarray],
        batch_size: int = 4
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for each split.

        Note: Data is NOT shuffled to preserve temporal ordering.
        For time series, maintaining chronological order is important
        even at the sequence level.

        Args:
            normalized_splits: Dict of normalized data arrays
            batch_size: Batch size for training (small due to limited data)

        Returns:
            Dict with DataLoader for each split
        """
        dataloaders = {}

        for name, data in normalized_splits.items():
            if len(data) == 0:
                dataloaders[name] = None
                continue

            dataset = TimeSeriesDataset(data)

            # No shuffling - preserve temporal ordering
            loader = DataLoader(
                dataset,
                batch_size=min(batch_size, len(data)),
                shuffle=False,
                drop_last=False,
            )

            dataloaders[name] = loader
            logger.debug(f"Created DataLoader for {name}: {len(dataset)} samples")

        return dataloaders

    def get_test_week_info(self) -> List[Dict]:
        """Get metadata for test weeks (useful for evaluation)."""
        if self.week_info is None or not hasattr(self, "split_indices"):
            return []

        return [self.week_info[i] for i in self.split_indices.get("test", [])]

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Args:
            data: Normalized data array

        Returns:
            Data in original scale
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call normalize() first.")

        original_shape = data.shape
        flat = data.flatten().reshape(-1, 1)
        original = self.scaler.inverse_transform(flat)
        return original.reshape(original_shape)

    def preprocess(
        self,
        filepath: str,
        batch_size: int = 4
    ) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray]]:
        """
        Run the complete preprocessing pipeline.

        This is the main entry point that runs all preprocessing steps:
        1. Load data
        2. Segment into weeks
        3. Create splits
        4. Normalize
        5. Create DataLoaders

        Args:
            filepath: Path to nyc_taxi.csv
            batch_size: Batch size for DataLoaders

        Returns:
            Tuple of (dataloaders_dict, normalized_splits_dict)
        """
        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 60)

        # Step 1: Load data
        self.load_data(filepath)

        # Step 2: Segment into weeks
        self.segment_into_weeks()

        # Step 3: Create splits
        splits = self.create_splits()

        # Step 4: Normalize
        normalized_splits = self.normalize(splits)

        # Step 5: Create DataLoaders
        dataloaders = self.create_dataloaders(normalized_splits, batch_size=batch_size)

        logger.info("=" * 60)
        logger.info("Preprocessing complete")
        logger.info("=" * 60)

        return dataloaders, normalized_splits

    def get_stats(self) -> Dict:
        """Get preprocessing statistics."""
        stats = {
            "config": {
                "sequence_length": self.config.sequence_length,
                "train_weeks": self.config.train_weeks,
                "val_weeks": self.config.val_weeks,
                "threshold_weeks": self.config.threshold_weeks,
            },
            "data_loaded": self.raw_df is not None,
            "weeks_segmented": self.weekly_data is not None,
            "scaler_fitted": self.scaler is not None,
        }

        if self.raw_df is not None:
            stats["total_records"] = len(self.raw_df)
            stats["date_range"] = {
                "start": str(self.raw_df["timestamp"].min()),
                "end": str(self.raw_df["timestamp"].max()),
            }

        if self.weekly_data is not None:
            stats["total_weeks"] = len(self.weekly_data)
            stats["normal_weeks"] = sum(
                1 for w in self.week_info if not w["is_anomaly"]
            )
            stats["anomaly_weeks"] = sum(
                1 for w in self.week_info if w["is_anomaly"]
            )

        if self.scaler is not None:
            stats["scaler"] = {
                "mean": float(self.scaler.mean_[0]),
                "std": float(self.scaler.scale_[0]),
            }

        return stats


def main():
    """Test the preprocessor with the NYC taxi dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess NYC taxi data")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/nyc_taxi.csv",
        help="Path to NYC taxi CSV file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoaders"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Run preprocessing
    preprocessor = NYCTaxiPreprocessor()
    dataloaders, normalized_splits = preprocessor.preprocess(
        args.data_path,
        batch_size=args.batch_size
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    stats = preprocessor.get_stats()
    print(f"\nTotal records: {stats.get('total_records', 'N/A')}")
    print(f"Total complete weeks: {stats.get('total_weeks', 'N/A')}")
    print(f"Normal weeks: {stats.get('normal_weeks', 'N/A')}")
    print(f"Weeks with anomalies: {stats.get('anomaly_weeks', 'N/A')}")

    print("\nData splits:")
    for name, loader in dataloaders.items():
        if loader is not None:
            print(f"  {name}: {len(loader.dataset)} weeks, {len(loader)} batches")
        else:
            print(f"  {name}: empty")

    print("\nNormalization:")
    if "scaler" in stats:
        print(f"  Mean: {stats['scaler']['mean']:.2f}")
        print(f"  Std: {stats['scaler']['std']:.2f}")

    # Verify tensor shapes
    print("\nTensor shapes:")
    for name, loader in dataloaders.items():
        if loader is not None:
            sample = next(iter(loader))
            print(f"  {name}: {sample.shape} (batch, seq_len, features)")
            break

    print("\nTest weeks with anomalies:")
    for week in preprocessor.get_test_week_info():
        if week["is_anomaly"]:
            print(f"  {week['year_week']}: {week['start_date'].date()} to {week['end_date'].date()}")


if __name__ == "__main__":
    main()
