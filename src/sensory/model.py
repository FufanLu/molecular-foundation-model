"""3D molecular encoder with sensory-label prototype alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSensoryModel(nn.Module):
    """Predict sensory labels and align a 3D molecule space to label prototypes.

    Odor and taste are not independent views of the same molecular embedding.
    Instead, molecules are mapped to one shared projection space, while odor
    and taste labels each own prototypes in that space. Cross-sensory losses
    therefore align label sets supported by the same molecule, avoiding the
    trivial same-molecule identity objective of separate projection heads.
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
        # FlavorDB wording remains a separate, optional weak-supervision head.
        self.weak_taste_head = nn.Linear(embedding_dim, taste_dim)
        self.molecule_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.odor_prototypes = nn.Parameter(torch.empty(odor_dim, projection_dim))
        self.taste_prototypes = nn.Parameter(torch.empty(taste_dim, projection_dim))
        nn.init.normal_(self.odor_prototypes, std=0.02)
        nn.init.normal_(self.taste_prototypes, std=0.02)

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
        return {
            "embedding": embedding,
            "odor_logits": self.odor_head(embedding),
            "taste_logits": self.taste_head(embedding),
            "weak_taste_logits": self.weak_taste_head(embedding),
            "molecule_projection": F.normalize(self.molecule_projection(embedding), dim=-1),
            "odor_prototypes": F.normalize(self.odor_prototypes, dim=-1),
            "taste_prototypes": F.normalize(self.taste_prototypes, dim=-1),
        }
