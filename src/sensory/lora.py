"""Device-safe LoRA adapters used by the cross-sensory Uni-Mol model."""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Freeze a linear layer and add a trainable low-rank residual."""

    def __init__(self, original: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.original = original
        for parameter in self.original.parameters():
            parameter.requires_grad = False
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.zeros(r, original.in_features, device=original.weight.device, dtype=original.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(original.out_features, r, device=original.weight.device, dtype=original.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = (self.dropout(inputs) @ self.lora_A.T) @ self.lora_B.T
        return self.original(inputs) + self.scaling * residual


def apply_lora(
    model: nn.Module,
    r: int = 4,
    alpha: int = 8,
    dropout: float = 0.1,
    target_names: list[str] | None = None,
) -> int:
    """Replace matching linear modules and return the replacement count."""
    target_names = target_names or ["q_proj", "k_proj", "v_proj"]
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not any(name.endswith(target) for target in target_names):
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced += 1
    return replaced
