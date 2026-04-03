"""Sparse Autoencoder (SAE) implementation for mechanistic interpretability.

Implements a sparse autoencoder with tied encoder-decoder weights,
ReLU activation, and L1 sparsity penalty for interpretable feature learning.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SparseAutoencoder(nn.Module):  # type: ignore[misc]
    """Sparse autoencoder for learning interpretable latent features.

    Architecture:
            Input (896) → Encoder → Latent (3584) → Decoder → Reconstruction (896)

    Features:
            - Tied weights: decoder weights = encoder weights transposed
            - ReLU activation: enforces non-negativity
            - L1 penalty: encourages sparsity (most neurons inactive)
    """

    def __init__(self, input_dim: int = 896, latent_dim: int = 3584) -> None:
        """Initialize the sparse autoencoder.

        Args:
                input_dim: Dimension of activation vectors from the LLM.
                latent_dim: Dimension of the latent feature space. Should be
                        larger than input_dim for the SAE setting.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder parameters
        self.encoder_weight = nn.Parameter(torch.empty(input_dim, latent_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(latent_dim))

        # Decoder bias only; decoder weights are tied to encoder weights
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform and zeros."""
        nn.init.xavier_uniform_(self.encoder_weight)
        nn.init.zeros_(self.encoder_bias)
        nn.init.zeros_(self.decoder_bias)

    def to_device_and_dtype(self, x: Tensor) -> Tensor:
        """Move inputs to the SAE's device and dtype.

        Args:
                x: Input tensor to cast and move.

        Returns:
                Tensor on the same device and dtype as the encoder weights.
        """
        weight = self.encoder_weight
        if x.device == weight.device and x.dtype == weight.dtype:
            return x
        return x.to(device=weight.device, dtype=weight.dtype)

    def encode(self, x: Tensor) -> Tensor:
        """Encode input activations into sparse latent features.

        Args:
                x: Input tensor of shape (batch_size, input_dim).

        Returns:
                Encoded latent tensor of shape (batch_size, latent_dim).
        """
        x = self.to_device_and_dtype(x)
        z_pre = x @ self.encoder_weight + self.encoder_bias
        z = F.relu(z_pre)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent features back to activation space.

        Uses tied weights: decoder_weight = encoder_weight.T

        Args:
                z: Latent tensor of shape (batch_size, latent_dim).

        Returns:
                Reconstructed activation tensor of shape (batch_size, input_dim).
        """
        return z @ self.encoder_weight.T + self.decoder_bias

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass through full SAE (encode then decode).

        Args:
                x: Input tensor of shape (batch_size, input_dim).

        Returns:
                                Dictionary with keys:
                                                - "latent": encoded features of shape
                                                    (batch_size, latent_dim)
                                                - "reconstruction": decoded activations
                                                    of shape
                                                    (batch_size, input_dim)
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)

        return {
            "latent": z,
            "reconstruction": x_reconstructed,
        }

    def compute_loss(self, x: Tensor, l1_lambda: float) -> dict[str, Tensor]:
        """Compute reconstruction loss + L1 sparsity penalty.

        Loss = MSE(reconstruction, input) + lambda * mean(|latent|)

        Args:
                x: Input tensor of shape (batch_size, input_dim).
                l1_lambda: Weight for L1 sparsity penalty on latent activations.

        Returns:
                Dictionary with keys:
                        - "loss": total loss (scalar)
                        - "reconstruction_loss": MSE loss (scalar)
                        - "l1_loss": mean absolute latent values (scalar)
                                                - "latent": encoded features of shape
                                                    (batch_size, latent_dim)
                                                - "reconstruction": decoded output
                                                    of shape
                                                    (batch_size, input_dim)
        """
        outputs = self.forward(x)
        z = outputs["latent"]
        x_reconstructed = outputs["reconstruction"]

        reconstruction_loss = F.mse_loss(x_reconstructed, self.to_device_and_dtype(x))
        l1_loss = z.abs().mean()
        total_loss = reconstruction_loss + l1_lambda * l1_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "l1_loss": l1_loss,
            "latent": z,
            "reconstruction": x_reconstructed,
        }

    def state_dict_for_checkpoint(self) -> dict[str, Any]:
        """Build a checkpoint-friendly state dict payload.

        Returns:
                Dictionary with model hyperparameters and state dict.
        """
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "model_state_dict": self.state_dict(),
        }
