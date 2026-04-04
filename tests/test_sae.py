"""Unit tests for SparseAutoencoder and SAE loss behavior."""

import torch

from brain_surgery.sae import SparseAutoencoder


def test_projection_896_to_3584(sae_fixture: SparseAutoencoder) -> None:
    """Verify encoder projects 896-dim activations to 3584 latents."""
    x = torch.randn(5, 896)
    z = sae_fixture.encode(x)
    assert z.shape == (5, 3584)


def test_decode_uses_tied_weights(sae_fixture: SparseAutoencoder) -> None:
    """Verify reconstruction path uses encoder_weight.T tied-weight rule."""
    z = torch.randn(4, 3584)
    decoded = sae_fixture.decode(z)
    expected = z @ sae_fixture.encoder_weight.T + sae_fixture.decoder_bias
    assert decoded.shape == (4, 896)
    assert torch.allclose(decoded, expected)


def test_l1_loss_positive_and_contributes(sae_fixture: SparseAutoencoder) -> None:
    """Verify L1 term is positive and contributes to total loss."""
    x = torch.rand(8, 896)  # non-negative to keep ReLU latents active
    losses = sae_fixture.compute_loss(x, l1_lambda=1e-3)

    recon = losses["reconstruction_loss"]
    l1 = losses["l1_loss"]
    total = losses["loss"]

    assert float(l1.item()) > 0.0
    assert float(total.item()) > float(recon.item())
