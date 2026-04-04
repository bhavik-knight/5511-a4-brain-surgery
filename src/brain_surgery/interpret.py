"""Feature interpretation and ranking for learned SAE features.

Provides methods to identify and analyze which inputs activate
specific latent features, enabling feature interpretation.
"""

from pathlib import Path

import torch
from torch import Tensor

from .sae import SparseAutoencoder

type MetadataValue = int | float | str | None
type MetadataRow = dict[str, MetadataValue]


class SAEInterpreter:
    """Interpreter for learned sparse autoencoder features.

    Ranks and analyzes which input tokens most strongly activate
    each latent feature to discover feature semantics.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        dataset_path: Path,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the SAE interpreter.

        Args:
                checkpoint_path: Path to SAE model checkpoint.
                dataset_path: Path to activation dataset .pt file.
                device: Device to run SAE on. Defaults to "cpu".
        """
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.device = device

        self.model: SparseAutoencoder | None = None
        self.activation_matrix: Tensor | None = None
        self.metadata: list[MetadataRow] | None = None
        self.latents: Tensor | None = None

    def load(self) -> None:
        """Load SAE model, activations, and metadata from disk.

        Raises:
                ValueError: If metadata and activation matrix counts don't match.
        """
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        input_dim = checkpoint["input_dim"]
        latent_dim = checkpoint["latent_dim"]

        self.model = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        dataset = torch.load(self.dataset_path, map_location="cpu")
        self.activation_matrix = dataset["activation_matrix"]
        self.metadata = dataset["metadata"]

        if self.metadata is None or self.activation_matrix is None:
            raise ValueError("Dataset is missing activation_matrix or metadata")
        if len(self.metadata) != self.activation_matrix.shape[0]:
            raise ValueError(
                f"Metadata length ({len(self.metadata)}) does not match "
                f"number of activation rows ({self.activation_matrix.shape[0]})."
            )

    def compute_latents(self) -> Tensor:
        """Compute latent features for all activations.

        Returns:
                Latent feature matrix of shape (num_tokens, latent_dim).

        Raises:
                RuntimeError: If load() was not called first.
        """
        if self.model is None or self.activation_matrix is None:
            raise RuntimeError("Call load() before compute_latents().")

        with torch.no_grad():
            self.latents = self.model.encode(
                self.activation_matrix.to(self.device)
            ).cpu()

        return self.latents

    def get_top_examples_for_feature(
        self,
        feature_index: int,
        top_k: int = 10,
    ) -> list[MetadataRow]:
        """Get tokens most strongly activating a specific feature.

        Finds the top-k token activations for a given feature across
        the entire dataset.

        Args:
                feature_index: Index of the latent feature.
                top_k: Number of top examples to return. Defaults to 10.

        Returns:
                List of dictionaries with keys:
                        - "rank": Ranking (1 = highest activation)
                        - "feature_value": Activation strength
                        - "token_text": The token string
                        - "prompt_text": Source prompt
                        - Other metadata fields

        Raises:
                RuntimeError: If compute_latents() not called first.
                ValueError: If feature_index out of range.
        """
        if self.latents is None or self.metadata is None:
            raise RuntimeError(
                "Call compute_latents() before get_top_examples_for_feature()."
            )

        if feature_index < 0 or feature_index >= self.latents.shape[1]:
            raise ValueError(
                f"feature_index {feature_index} is out of range. "
                f"Valid range: 0 to {self.latents.shape[1] - 1}."
            )

        feature_values = self.latents[:, feature_index]
        top_values, top_indices = torch.topk(
            feature_values, k=min(top_k, len(feature_values))
        )

        results: list[MetadataRow] = []
        for rank, (row_idx, value) in enumerate(
            zip(top_indices.tolist(), top_values.tolist()), start=1
        ):
            meta = self.metadata[row_idx]
            results.append(
                {
                    "rank": rank,
                    "row_index": row_idx,
                    "feature_index": feature_index,
                    "feature_value": float(value),
                    "prompt_id": meta.get("prompt_id"),
                    "prompt_text": meta.get("prompt_text"),
                    "token_index": meta.get("token_index"),
                    "token_id": meta.get("token_id"),
                    "token_text": meta.get("token_text"),
                    "token_str": meta.get("token_str"),
                    "generated_text": meta.get("generated_text"),
                    "hook_layer_index": meta.get("hook_layer_index"),
                    "hook_layer_name": meta.get("hook_layer_name"),
                }
            )

        return results

    def get_top_features_for_row(
        self,
        row_index: int,
        top_k: int = 10,
    ) -> list[MetadataRow]:
        """Get most strongly activated features for a specific token.

        Args:
                row_index: Row index in the activation matrix (token index).
                top_k: Number of top features to return. Defaults to 10.

        Returns:
                List of dictionaries with keys:
                        - "rank": Ranking (1 = strongest activation)
                        - "feature_index": Index of the latent feature
                        - "feature_value": Activation strength
                        - "token_text": The token string

        Raises:
                RuntimeError: If compute_latents() not called first.
                ValueError: If row_index out of range.
        """
        if self.latents is None or self.metadata is None:
            raise RuntimeError(
                "Call compute_latents() before get_top_features_for_row()."
            )

        if row_index < 0 or row_index >= self.latents.shape[0]:
            raise ValueError(
                f"row_index {row_index} is out of range. "
                f"Valid range: 0 to {self.latents.shape[0] - 1}."
            )

        row_values = self.latents[row_index]
        top_values, top_indices = torch.topk(row_values, k=min(top_k, len(row_values)))

        meta = self.metadata[row_index]
        results: list[MetadataRow] = []
        for rank, (feature_idx, value) in enumerate(
            zip(top_indices.tolist(), top_values.tolist()), start=1
        ):
            results.append(
                {
                    "rank": rank,
                    "row_index": row_index,
                    "feature_index": int(feature_idx),
                    "feature_value": float(value),
                    "prompt_text": meta.get("prompt_text"),
                    "token_index": meta.get("token_index"),
                    "token_text": meta.get("token_text"),
                }
            )

        return results

    def rank_features_by_max_activation(self, top_k: int = 20) -> list[MetadataRow]:
        """Rank features by their single strongest activation.

        Finds features that maximally activate on specific tokens,
        useful for discovering feature semantics.

        Args:
                top_k: Number of top features to return. Defaults to 20.

        Returns:
                List of dictionaries with keys:
                        - "rank": Feature ranking
                        - "feature_index": Index of the feature
                        - "max_feature_value": Strongest activation value
                        - "token_text": Token that maximally activated it
                        - "prompt_text": Source prompt

        Raises:
                RuntimeError: If compute_latents() not called first.
        """
        if self.latents is None or self.metadata is None:
            raise RuntimeError(
                "Call compute_latents() before rank_features_by_max_activation()."
            )

        max_values, max_rows = torch.max(self.latents, dim=0)
        top_feature_values, top_feature_indices = torch.topk(
            max_values,
            k=min(top_k, self.latents.shape[1]),
        )

        results: list[MetadataRow] = []
        for rank, (feature_idx, feature_value) in enumerate(
            zip(top_feature_indices.tolist(), top_feature_values.tolist()), start=1
        ):
            row_index = int(max_rows[feature_idx].item())
            meta = self.metadata[row_index]
            results.append(
                {
                    "rank": rank,
                    "feature_index": int(feature_idx),
                    "max_feature_value": float(feature_value),
                    "row_index_of_max": row_index,
                    "token_text": meta.get("token_text"),
                    "token_index": meta.get("token_index"),
                    "prompt_text": meta.get("prompt_text"),
                }
            )

        return results
