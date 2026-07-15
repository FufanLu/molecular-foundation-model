"""LoRA adapter + MLP head for Uni-Mol odor classification."""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, original, r=4, alpha=8, dropout=0.1):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Create adapters on the same device/dtype as the wrapped layer.
        # This matters when LoRA is injected after the backbone has already
        # been moved to CUDA (as happens in the Colab notebook).
        self.lora_A = nn.Parameter(
            torch.zeros(
                r,
                original.in_features,
                device=original.weight.device,
                dtype=original.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                original.out_features,
                r,
                device=original.weight.device,
                dtype=original.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.original(x)
        delta = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return result + self.scaling * delta


def apply_lora(model, r=4, alpha=8, dropout=0.1, target_names=None):
    if target_names is None:
        target_names = ["q_proj", "k_proj", "v_proj"]

    replaced = 0
    for name, module in model.named_modules():
        for target in target_names:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent = model
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced


class OdorClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim=128, num_classes=5, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch):
        outputs = self.backbone(**batch)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)

    def predict(self, batch):
        logits = self.forward(batch)
        return torch.sigmoid(logits)
