"""Shared Uni-Mol encoder with odor/taste heads in one latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSensoryModel(nn.Module):
    """Predict odor and taste while aligning their molecule representations.

    The encoder is shared.  Separate task heads prevent one modality's label
    convention from overwriting the other, while the two projection heads are
    aligned only for molecules that have labels in both modalities.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        odor_dim: int,
        taste_dim: int,
        projection_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )
        self.odor_head = nn.Linear(embedding_dim, odor_dim)
        self.taste_head = nn.Linear(embedding_dim, taste_dim)
        self.odor_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.taste_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get the differentiable molecular CLS representation from Uni-Mol."""
        outputs = self.backbone(**batch, return_repr=True)
        if isinstance(outputs, dict):
            if "cls_repr" in outputs:
                return outputs["cls_repr"]
            if "atomic_reprs" in outputs:
                return outputs["atomic_reprs"][:, 0, :]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0, :]
        if torch.is_tensor(outputs):
            return outputs
        raise TypeError("Could not extract a molecular representation from the Uni-Mol backbone.")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = self.trunk(self.encode(batch))
        odor_projection = F.normalize(self.odor_projection(embedding), dim=-1)
        taste_projection = F.normalize(self.taste_projection(embedding), dim=-1)
        return {
            "embedding": embedding,
            "odor_logits": self.odor_head(embedding),
            "taste_logits": self.taste_head(embedding),
            "odor_projection": odor_projection,
            "taste_projection": taste_projection,
        }
