"""Models, losses, and evaluation utilities for cross-sensory learning."""

from .losses import (
    cross_sensory_loss,
    label_set_embeddings,
    masked_bce_with_logits,
    molecule_prototype_nce,
    paired_label_set_info_nce,
)
from .lora import LoRALinear, apply_lora
from .metrics import compute_pos_weight, macro_f1, profile_retrieval, select_thresholds
from .model import CrossSensoryModel

__all__ = [
    "CrossSensoryModel",
    "LoRALinear",
    "apply_lora",
    "compute_pos_weight",
    "cross_sensory_loss",
    "label_set_embeddings",
    "macro_f1",
    "masked_bce_with_logits",
    "molecule_prototype_nce",
    "paired_label_set_info_nce",
    "profile_retrieval",
    "select_thresholds",
]
