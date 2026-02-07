"""
Credit Card Fraud Detection Data Preprocessor

Handles data loading, feature transformation, stratified splitting, normalization,
and DataLoader creation for credit card transaction data.

Pattern adapted from data_preprocessor.py but for tabular data instead of time series.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class CreditCardPreprocessorConfig:
    """Configuration for credit card data preprocessing."""

    train_ratio: float = 0.7           # 70% for train/test split
    val_ratio_of_train: float = 0.2    # 20% of training for validation
    random_seed: int = 42              # For reproducibility
    normalization_range: tuple = field(default_factory=lambda: (-1, 1))  # Min-max range
    log_transform_amount: bool = True  # Apply log transform to Amount
    use_validation_split: bool = True  # Split training data into train/val


class CreditCardDataset(Dataset):
    """
    PyTorch Dataset for credit card transactions.

    Unlike TimeSeriesDataset (which handles sequences), this handles individual
    transaction feature vectors.
    """

    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Args:
            features: Transaction features, shape (num_transactions, num_features)
            labels: Fraud labels (0=legitimate, 1=fraud), shape (num_transactions,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class CreditCardPreprocessor:
    """
    Preprocessor for credit card fraud detection data.

    Pipeline:
    1. Load CSV and verify structure
    2. Transform Time feature (seconds → hour of day)
    3. Transform Amount feature (apply log transform)
    4. Create stratified train/test/val splits
    5. Normalize features to [-1, 1] using MinMaxScaler
    6. Create PyTorch DataLoaders
    """

    def __init__(self, config: Optional[CreditCardPreprocessorConfig] = None):
        """
        Args:
            config: Preprocessing configuration
        """
        self.config = config or CreditCardPreprocessorConfig()
        self.scaler: Optional[MinMaxScaler] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.feature_names: Optional[list] = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and validate credit card CSV file.

        Args:
            filepath: Path to creditcard.csv

        Returns:
            DataFrame with all transactions

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)

        # Verify required columns
        required_cols = ['Time', 'Amount', 'Class']
        pca_cols = [f'V{i}' for i in range(1, 29)]
        all_required = required_cols + pca_cols

        missing = set(all_required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Log dataset statistics
        total = len(df)
        fraud_count = (df['Class'] == 1).sum()
        fraud_ratio = fraud_count / total * 100

        logger.info(f"Loaded {total:,} transactions")
        logger.info(f"Fraud transactions: {fraud_count:,} ({fraud_ratio:.3f}%)")
        logger.info(f"Legitimate transactions: {total - fraud_count:,} ({100 - fraud_ratio:.3f}%)")
        logger.info(f"Time range: {df['Time'].min():.0f}s - {df['Time'].max():.0f}s")
        logger.info(f"Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")

        self.raw_df = df
        return df

    def transform_time_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform Time (seconds since first transaction) to hour of day.

        This captures daily cyclical patterns (fraud may peak at certain hours)
        while discarding absolute position in the dataset.

        Formula: Hour = floor((Time % 86400) / 3600)

        Args:
            df: DataFrame with Time column

        Returns:
            DataFrame with Hour column (Time removed)
        """
        df = df.copy()

        # Convert seconds to hour of day (0-23)
        df['Hour'] = ((df['Time'] % 86400) // 3600).astype(int)
        df = df.drop('Time', axis=1)

        logger.info("Transformed Time → Hour (0-23)")
        logger.info(f"Hour distribution: {df['Hour'].value_counts().sort_index().to_dict()}")

        return df

    def transform_amount_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transform to Amount feature.

        Amount has extreme right skew (max 25,691). Log transform reduces outlier
        impact and makes the distribution more normal.

        Formula: Amount_log = log(1 + Amount)

        Args:
            df: DataFrame with Amount column

        Returns:
            DataFrame with Amount_log column (Amount removed)
        """
        df = df.copy()

        # Log1p handles Amount=0 gracefully
        df['Amount_log'] = np.log1p(df['Amount'])
        df = df.drop('Amount', axis=1)

        logger.info("Transformed Amount → Amount_log")
        logger.info(f"Amount_log range: {df['Amount_log'].min():.3f} - {df['Amount_log'].max():.3f}")

        return df

    def create_splits(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified train/val/test splits.

        Split strategy (with validation):
        1. 70/30 train/test split (stratified by fraud ratio)
        2. 80/20 train/val split of training data (stratified)

        Final distribution (with validation):
        - Train: 56% of total (0.7 * 0.8)
        - Val: 14% of total (0.7 * 0.2)
        - Test: 30% of total

        Final distribution (without validation):
        - Train: 70% of total
        - Test: 30% of total

        All splits preserve the ~0.172% fraud ratio.

        Args:
            df: DataFrame with features and Class label

        Returns:
            Dict with keys 'train', 'val' (optional), 'test', values (X, y) tuples
        """
        # Separate features and labels
        feature_cols = [col for col in df.columns if col != 'Class']
        X = df[feature_cols].values.astype(np.float32)
        y = df['Class'].values.astype(np.float32)

        # Save feature names for later reference
        self.feature_names = feature_cols

        # 70/30 train/test split (stratified)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=(1 - self.config.train_ratio),
            stratify=y,
            random_state=self.config.random_seed
        )

        if self.config.use_validation_split:
            # 80/20 train/val split of training data (stratified)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=self.config.val_ratio_of_train,
                stratify=y_train_full,
                random_state=self.config.random_seed
            )

            # Log split statistics
            logger.info(f"\nData splits created (with validation):")
            logger.info(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
            logger.info(f"    Fraud: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.3f}%)")
            logger.info(f"  Val: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
            logger.info(f"    Fraud: {(y_val == 1).sum():,} ({(y_val == 1).mean()*100:.3f}%)")
            logger.info(f"  Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
            logger.info(f"    Fraud: {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.3f}%)")

            return {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
        else:
            # Use full 70% for training (no validation split)
            X_train = X_train_full
            y_train = y_train_full

            # Log split statistics
            logger.info(f"\nData splits created (NO validation - full 70% for training):")
            logger.info(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
            logger.info(f"    Fraud: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.3f}%)")
            logger.info(f"  Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
            logger.info(f"    Fraud: {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.3f}%)")

            return {
                'train': (X_train, y_train),
                'test': (X_test, y_test)
            }

    def normalize(
        self,
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Normalize features to [-1, 1] using MinMaxScaler.

        CRITICAL: Scaler is fit ONLY on training data to prevent data leakage.

        Args:
            splits: Dict with (X, y) tuples for train/val/test

        Returns:
            Dict with normalized (X, y) tuples
        """
        X_train, y_train = splits['train']

        # Initialize and fit scaler on training data only
        self.scaler = MinMaxScaler(feature_range=self.config.normalization_range)
        self.scaler.fit(X_train)

        logger.info(f"\nFitted MinMaxScaler on training data:")
        logger.info(f"  Feature range: {self.config.normalization_range}")
        logger.info(f"  Data min: {self.scaler.data_min_[:5]}... (first 5 features)")
        logger.info(f"  Data max: {self.scaler.data_max_[:5]}... (first 5 features)")

        # Transform all splits using training statistics
        normalized = {}
        for name, (X, y) in splits.items():
            X_normalized = self.scaler.transform(X)
            normalized[name] = (X_normalized, y)

            logger.info(f"  {name.capitalize()}: normalized to [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")

        return normalized

    def create_dataloaders(
        self,
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 256
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for each split.

        Unlike time series DataLoaders (which never shuffle), we shuffle
        training data for better gradient estimates.

        Args:
            splits: Dict with normalized (X, y) tuples
            batch_size: Batch size (default: 256, larger than LSTM's 4 due to more data)

        Returns:
            Dict with DataLoader for each split
        """
        dataloaders = {}

        for name, (X, y) in splits.items():
            dataset = CreditCardDataset(X, y)

            # Shuffle train only (val/test kept in order for reproducibility)
            shuffle = (name == 'train')

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,  # Use all data
                num_workers=0  # Single-threaded for compatibility
            )

            dataloaders[name] = loader
            logger.info(f"  {name.capitalize()}: {len(dataset):,} samples, {len(loader)} batches, shuffle={shuffle}")

        return dataloaders

    def preprocess(
        self,
        filepath: str,
        batch_size: int = 256
    ) -> Tuple[Dict[str, DataLoader], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Main entry point: complete preprocessing pipeline.

        Steps:
        1. Load data
        2. Transform Time → Hour
        3. Transform Amount → Amount_log (if enabled)
        4. Create stratified splits
        5. Normalize
        6. Create DataLoaders

        Args:
            filepath: Path to creditcard.csv
            batch_size: Batch size for DataLoaders

        Returns:
            (dataloaders, normalized_splits)
        """
        logger.info("=" * 80)
        logger.info("Credit Card Data Preprocessing Pipeline")
        logger.info("=" * 80)

        # Step 1: Load data
        df = self.load_data(filepath)

        # Step 2: Transform Time
        df = self.transform_time_feature(df)

        # Step 3: Transform Amount (optional)
        if self.config.log_transform_amount:
            df = self.transform_amount_feature(df)

        # Step 4: Create splits
        splits = self.create_splits(df)

        # Step 5: Normalize
        normalized_splits = self.normalize(splits)

        # Step 6: Create DataLoaders
        dataloaders = self.create_dataloaders(normalized_splits, batch_size)

        logger.info("=" * 80)
        logger.info("Preprocessing complete!")
        logger.info("=" * 80)

        return dataloaders, normalized_splits

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Useful for visualizing reconstructions.

        Args:
            data: Normalized data array

        Returns:
            Data in original scale

        Raises:
            ValueError: If scaler not fitted
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call normalize() first.")

        return self.scaler.inverse_transform(data)


def main():
    """Test the preprocessor on credit card data."""
    import argparse

    parser = argparse.ArgumentParser(description="Test credit card preprocessor")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/creditcard.csv",
        help="Path to credit card CSV file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for DataLoaders"
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create preprocessor and run pipeline
    config = CreditCardPreprocessorConfig()
    preprocessor = CreditCardPreprocessor(config=config)

    dataloaders, splits = preprocessor.preprocess(
        filepath=args.data_path,
        batch_size=args.batch_size
    )

    # Test loading a batch
    logger.info("\nTesting DataLoader:")
    train_loader = dataloaders['train']
    features, labels = next(iter(train_loader))
    logger.info(f"  Batch features shape: {features.shape}")
    logger.info(f"  Batch labels shape: {labels.shape}")
    logger.info(f"  Features range: [{features.min():.3f}, {features.max():.3f}]")
    logger.info(f"  Labels: {labels.unique()}")


if __name__ == "__main__":
    main()
