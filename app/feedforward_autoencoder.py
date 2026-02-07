"""
Feedforward Autoencoder for Credit Card Fraud Detection

Implements a simple feedforward encoder-decoder architecture:
- Encoder: 30 → d1 → 15 (compress to latent space)
- Decoder: 15 → d1 → 30 (reconstruct from latent space)

Pattern adapted from lstm_autoencoder.py but simplified to feedforward layers.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AutoencoderConfig:
    """Configuration for feedforward autoencoder."""

    input_dim: int = 30       # V1-V28 + Hour + Amount_log
    hidden_dim: int = 25      # d1 (intermediate dimension)
    latent_dim: int = 10      # Bottleneck dimension
    dropout: float = 0.0      # Dropout rate (0 = no dropout)


class Encoder(nn.Module):
    """
    Two-layer feedforward encoder: 30 → d1 → 15

    Compresses input transaction vector to latent representation.
    """

    def __init__(self, config: AutoencoderConfig):
        """
        Args:
            config: Autoencoder configuration
        """
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.latent_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor, shape (batch, input_dim)

        Returns:
            z: Latent representation, shape (batch, latent_dim)
        """
        # First layer: input → hidden
        h = F.relu(self.fc1(x))
        h = self.dropout(h)

        # Second layer: hidden → latent
        z = F.relu(self.fc2(h))

        return z


class Decoder(nn.Module):
    """
    Two-layer feedforward decoder: 15 → d1 → 30

    Reconstructs input from latent representation.
    Mirrors the encoder architecture.
    """

    def __init__(self, config: AutoencoderConfig):
        """
        Args:
            config: Autoencoder configuration
        """
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.latent_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.input_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent representation, shape (batch, latent_dim)

        Returns:
            x_reconstructed: Reconstruction, shape (batch, input_dim)
        """
        # First layer: latent → hidden
        h = F.relu(self.fc1(z))
        h = self.dropout(h)

        # Second layer: hidden → output (linear activation)
        # Linear output because data is normalized to [-1, 1]
        x_reconstructed = self.fc2(h)

        return x_reconstructed


class FeedforwardAutoencoder(nn.Module):
    """
    Complete feedforward autoencoder for credit card fraud detection.

    Architecture: 30 → d1 → 15 → d1 → 30

    Key differences from EncDecAD (LSTM version):
    - No recurrent connections
    - No sequence reversing
    - No teacher forcing
    - Symmetric encoder-decoder architecture
    - Operates on individual transactions, not sequences
    """

    def __init__(self, config: Optional[AutoencoderConfig] = None):
        """
        Args:
            config: Autoencoder configuration (uses defaults if None)
        """
        super().__init__()
        self.config = config or AutoencoderConfig()

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

        logger.info(f"Created FeedforwardAutoencoder:")
        logger.info(f"  Architecture: {self.config.input_dim} → {self.config.hidden_dim} → "
                   f"{self.config.latent_dim} → {self.config.hidden_dim} → {self.config.input_dim}")
        logger.info(f"  Parameters: {self.count_parameters():,}")
        logger.info(f"  Dropout: {self.config.dropout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: Input tensor, shape (batch, input_dim)

        Returns:
            x_reconstructed: Reconstruction, shape (batch, input_dim)
        """
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation.

        Used in Phase 2 to generate features for MLP classifier.

        Args:
            x: Input tensor, shape (batch, input_dim)

        Returns:
            z: Latent representation, shape (batch, latent_dim)
        """
        return self.encoder(x)

    def count_parameters(self) -> int:
        """
        Count trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dict
        """
        return asdict(self.config)


class SemiSupervisedAutoencoder(nn.Module):
    """
    Autoencoder with a classification head on the latent representation.

    Adds a small linear probe to the bottleneck that predicts fraud/legitimate.
    Training uses a combined loss:
        L = L_reconstruction + λ * L_classification

    This encourages the latent representation to preserve fraud-relevant information
    that would otherwise be discarded by the pure reconstruction objective.
    """

    def __init__(self, autoencoder: FeedforwardAutoencoder):
        """
        Args:
            autoencoder: Base autoencoder to wrap
        """
        super().__init__()
        self.autoencoder = autoencoder
        self.config = autoencoder.config

        # Linear classification head on latent representation
        self.classifier = nn.Linear(self.config.latent_dim, 1)

        logger.info(f"Created SemiSupervisedAutoencoder:")
        logger.info(f"  Base autoencoder params: {autoencoder.count_parameters():,}")
        logger.info(f"  Classifier params: {sum(p.numel() for p in self.classifier.parameters()):,}")

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning both reconstruction and classification logit.

        Args:
            x: Input tensor, shape (batch, input_dim)

        Returns:
            (x_reconstructed, class_logit): Reconstruction and fraud probability logit
        """
        z = self.autoencoder.encoder(x)
        x_reconstructed = self.autoencoder.decoder(z)
        class_logit = self.classifier(z).squeeze(-1)  # (batch,)
        return x_reconstructed, class_logit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation."""
        return self.autoencoder.encode(x)

    @property
    def encoder(self):
        """Access the encoder for compatibility."""
        return self.autoencoder.encoder

    @property
    def decoder(self):
        """Access the decoder for compatibility."""
        return self.autoencoder.decoder

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_autoencoder(
    input_dim: int = 30,
    hidden_dim: int = 22,
    latent_dim: int = 15,
    dropout: float = 0.0
) -> FeedforwardAutoencoder:
    """
    Factory function to create autoencoder with custom configuration.

    Args:
        input_dim: Number of input features (default: 30)
        hidden_dim: Hidden layer dimension d1 (default: 22)
        latent_dim: Bottleneck dimension (default: 15)
        dropout: Dropout rate (default: 0.0)

    Returns:
        Configured FeedforwardAutoencoder
    """
    config = AutoencoderConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout
    )
    return FeedforwardAutoencoder(config=config)


def main():
    """Test the autoencoder architecture."""
    import argparse

    parser = argparse.ArgumentParser(description="Test feedforward autoencoder")
    parser.add_argument("--input-dim", type=int, default=30, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=22, help="Hidden dimension d1")
    parser.add_argument("--latent-dim", type=int, default=15, help="Latent dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for testing")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create model
    model = create_autoencoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim
    )

    # Test forward pass
    logger.info(f"\nTesting forward pass with batch size {args.batch_size}")
    x = torch.randn(args.batch_size, args.input_dim)
    logger.info(f"  Input shape: {x.shape}")

    # Full reconstruction
    x_reconstructed = model(x)
    logger.info(f"  Reconstruction shape: {x_reconstructed.shape}")

    # Latent encoding
    z = model.encode(x)
    logger.info(f"  Latent shape: {z.shape}")

    # Verify shapes
    assert x_reconstructed.shape == x.shape, "Reconstruction shape mismatch!"
    assert z.shape == (args.batch_size, args.latent_dim), "Latent shape mismatch!"

    logger.info("\n✓ All shape checks passed!")

    # Parameter breakdown
    logger.info("\nParameter breakdown:")
    for name, param in model.named_parameters():
        logger.info(f"  {name}: {param.shape} ({param.numel():,} params)")


if __name__ == "__main__":
    main()
