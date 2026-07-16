"""Masked supervision and evidence-tiered sensory-prototype objectives."""

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


def molecule_prototype_nce(
    molecule_projection: torch.Tensor,
    prototypes: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Align a molecule with every observed positive sensory-label prototype.

    Rows may have multiple positive labels. Their numerator is the log-sum-exp
    over all positive prototypes, while unknown labels are removed from the
    denominator instead of being silently treated as negatives.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if targets.shape[1] != prototypes.shape[0]:
        raise ValueError("Target and prototype label dimensions differ.")
    known = targets.ge(0)
    positive = targets.eq(1)
    active = known.any(dim=1) & positive.any(dim=1)
    if not active.any():
        return molecule_projection.sum() * 0.0
    logits = molecule_projection[active] @ prototypes.T / temperature
    known = known[active]
    positive = positive[active]
    negative_inf = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(~known, negative_inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, negative_inf), dim=1)
    return (denominator - numerator).mean()


def label_set_embeddings(prototypes: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean positive label prototypes for each row and return valid-row mask."""
    positive = targets.eq(1)
    valid = positive.any(dim=1)
    weights = positive.to(dtype=prototypes.dtype)
    pooled = weights @ prototypes / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return F.normalize(pooled, dim=-1), valid


def paired_label_set_info_nce(
    odor_prototypes: torch.Tensor,
    taste_prototypes: torch.Tensor,
    odor_targets: torch.Tensor,
    taste_targets: torch.Tensor,
    paired_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Align odor and taste label sets for evidence-supported molecules.

    The objective operates on label-set aggregates, not two projections of the
    same molecule, so it cannot be minimized simply by retaining molecular
    identity in parallel heads.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    odor_sets, odor_valid = label_set_embeddings(odor_prototypes, odor_targets)
    taste_sets, taste_valid = label_set_embeddings(taste_prototypes, taste_targets)
    selected = paired_mask.bool() & odor_valid & taste_valid
    if int(selected.sum()) < 2:
        return odor_prototypes.sum() * 0.0
    scores = odor_sets[selected] @ taste_sets[selected].T / temperature
    odor_positive = odor_targets[selected].eq(1)
    taste_positive = taste_targets[selected].eq(1)
    # Identical observed label sets describe the same sensory evidence. They
    # are multi-positives, not accidental negatives in another row's batch.
    same_odor_set = odor_positive.unsqueeze(1).eq(odor_positive.unsqueeze(0)).all(dim=-1)
    same_taste_set = taste_positive.unsqueeze(1).eq(taste_positive.unsqueeze(0)).all(dim=-1)
    positives = same_odor_set & same_taste_set
    negative_inf = torch.finfo(scores.dtype).min
    row_loss = torch.logsumexp(scores, dim=1) - torch.logsumexp(
        scores.masked_fill(~positives, negative_inf), dim=1
    )
    column_loss = torch.logsumexp(scores.T, dim=1) - torch.logsumexp(
        scores.T.masked_fill(~positives.T, negative_inf), dim=1
    )
    return 0.5 * (row_loss.mean() + column_loss.mean())


def cross_sensory_loss(
    outputs: dict[str, torch.Tensor],
    odor_targets: torch.Tensor,
    taste_targets: torch.Tensor,
    paired_mask: torch.Tensor,
    prototype_weight: float = 0.05,
    contrastive_weight: float = 0.05,
    temperature: float = 0.07,
    weak_taste_targets: torch.Tensor | None = None,
    weak_paired_mask: torch.Tensor | None = None,
    weak_taste_weight: float = 0.0,
    weak_contrastive_weight: float = 0.0,
    weak_temperature: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Combine task BCE with molecule--prototype and label-set alignment."""
    odor = masked_bce_with_logits(outputs["odor_logits"], odor_targets)
    taste = masked_bce_with_logits(outputs["taste_logits"], taste_targets)
    odor_prototype = molecule_prototype_nce(
        outputs["molecule_projection"], outputs["odor_prototypes"], odor_targets, temperature
    )
    taste_prototype = molecule_prototype_nce(
        outputs["molecule_projection"], outputs["taste_prototypes"], taste_targets, temperature
    )
    prototype = 0.5 * (odor_prototype + taste_prototype)
    strong_alignment = paired_label_set_info_nce(
        outputs["odor_prototypes"], outputs["taste_prototypes"],
        odor_targets, taste_targets, paired_mask, temperature,
    )
    weak_taste = (
        masked_bce_with_logits(outputs["weak_taste_logits"], weak_taste_targets)
        if weak_taste_targets is not None
        else outputs["weak_taste_logits"].sum() * 0.0
    )
    weak_alignment = (
        paired_label_set_info_nce(
            outputs["odor_prototypes"], outputs["taste_prototypes"],
            odor_targets, weak_taste_targets, weak_paired_mask, weak_temperature,
        )
        if weak_taste_targets is not None and weak_paired_mask is not None
        else outputs["taste_prototypes"].sum() * 0.0
    )
    total = (
        odor
        + taste
        + prototype_weight * prototype
        + contrastive_weight * strong_alignment
        + weak_taste_weight * weak_taste
        + weak_contrastive_weight * weak_alignment
    )
    return {
        "total": total,
        "odor": odor,
        "taste": taste,
        "prototype": prototype,
        "strong_alignment": strong_alignment,
        "weak_taste": weak_taste,
        "weak_alignment": weak_alignment,
    }
