"""Masked supervised and paired cross-modal contrastive objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE over observed labels only; targets equal to -1 are unknown."""
    if logits.shape != targets.shape:
        raise ValueError(f"Logit shape {tuple(logits.shape)} != target shape {tuple(targets.shape)}")
    mask = targets.ge(0)
    if not mask.any():
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits, targets.clamp_min(0), reduction="none")
    return losses.masked_select(mask).mean()


def paired_info_nce(
    odor_projection: torch.Tensor,
    taste_projection: torch.Tensor,
    paired_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE for exact molecule pairs present in the minibatch."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    selected = paired_mask.bool()
    odor = odor_projection[selected]
    taste = taste_projection[selected]
    if odor.shape[0] < 2:
        return odor_projection.sum() * 0.0
    scores = odor @ taste.T / temperature
    labels = torch.arange(scores.shape[0], device=scores.device)
    return 0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels))


def cross_sensory_loss(
    outputs: dict[str, torch.Tensor],
    odor_targets: torch.Tensor,
    taste_targets: torch.Tensor,
    paired_mask: torch.Tensor,
    contrastive_weight: float = 0.1,
    temperature: float = 0.07,
    weak_taste_targets: torch.Tensor | None = None,
    weak_paired_mask: torch.Tensor | None = None,
    weak_taste_weight: float = 0.0,
    weak_contrastive_weight: float = 0.0,
    weak_temperature: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Combine curated supervision with low-weight weak FlavorDB guidance."""
    odor = masked_bce_with_logits(outputs["odor_logits"], odor_targets)
    taste = masked_bce_with_logits(outputs["taste_logits"], taste_targets)
    contrastive = paired_info_nce(
        outputs["odor_projection"], outputs["taste_projection"], paired_mask, temperature
    )
    weak_taste = (
        masked_bce_with_logits(outputs["weak_taste_logits"], weak_taste_targets)
        if weak_taste_targets is not None
        else outputs["weak_taste_logits"].sum() * 0.0
    )
    weak_contrastive = (
        paired_info_nce(
            outputs["odor_projection"], outputs["weak_taste_projection"], weak_paired_mask, weak_temperature
        )
        if weak_paired_mask is not None
        else outputs["weak_taste_projection"].sum() * 0.0
    )
    total = (
        odor
        + taste
        + contrastive_weight * contrastive
        + weak_taste_weight * weak_taste
        + weak_contrastive_weight * weak_contrastive
    )
    return {
        "total": total,
        "odor": odor,
        "taste": taste,
        "strong_contrastive": contrastive,
        "weak_taste": weak_taste,
        "weak_contrastive": weak_contrastive,
    }
