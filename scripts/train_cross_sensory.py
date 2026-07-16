"""Colab entry point for scaffold-split cross-sensory Uni-Mol LoRA training.

Run this script in Colab after `python -m src.dataset.sensory ...` has created
`data/processed/sensory/molecules.parquet`.  It intentionally keeps DataHub
single-process: forking after CUDA initialisation is a common Colab deadlock.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, Sampler

from src.dataset.sensory import (
    LOW_SHOT_TASTE_LABELS,
    ODOR_FAMILIES,
    SCHEMA_VERSION,
    TASTE_LABELS,
    build_masked_targets,
)
from src.sensory import CrossSensoryModel, apply_lora, cross_sensory_loss


class SensoryDataset(Dataset):
    def __init__(
        self,
        features: list[dict[str, object]],
        odor_targets: np.ndarray,
        taste_targets: np.ndarray,
        weak_taste_targets: np.ndarray,
        strong_paired: np.ndarray,
        weak_only_paired: np.ndarray,
    ) -> None:
        self.features = features
        self.odor_targets = torch.from_numpy(odor_targets)
        self.taste_targets = torch.from_numpy(taste_targets)
        self.weak_taste_targets = torch.from_numpy(weak_taste_targets)
        self.strong_paired = torch.from_numpy(strong_paired.astype(np.bool_))
        self.weak_only_paired = torch.from_numpy(weak_only_paired.astype(np.bool_))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return (
            self.features[index], self.odor_targets[index], self.taste_targets[index],
            self.weak_taste_targets[index], self.strong_paired[index], self.weak_only_paired[index],
        )


class MultiPairBatchSampler(Sampler[list[int]]):
    """Place strong and weak-only molecule pairs in every contrastive batch."""

    def __init__(
        self,
        strong_paired: np.ndarray,
        weak_only_paired: np.ndarray,
        batch_size: int,
        strong_per_batch: int,
        weak_per_batch: int,
        seed: int,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least two")
        self.all_indices = np.arange(len(strong_paired), dtype=np.int64)
        self.strong_indices = self.all_indices[strong_paired.astype(bool)]
        self.weak_indices = self.all_indices[weak_only_paired.astype(bool)]
        self.batch_size = batch_size
        self.strong_per_batch = min(strong_per_batch, batch_size, len(self.strong_indices))
        available = batch_size - self.strong_per_batch
        self.weak_per_batch = min(weak_per_batch, available, len(self.weak_indices))
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.all_indices) / max(1, self.batch_size - self.strong_per_batch - self.weak_per_batch))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        all_indices = rng.permutation(self.all_indices)
        strong_indices = rng.permutation(self.strong_indices)
        weak_indices = rng.permutation(self.weak_indices)
        for batch_number in range(len(self)):
            strong = [
                int(strong_indices[(batch_number * self.strong_per_batch + offset) % len(strong_indices)])
                for offset in range(self.strong_per_batch)
            ] if self.strong_per_batch else []
            weak = [
                int(weak_indices[(batch_number * self.weak_per_batch + offset) % len(weak_indices)])
                for offset in range(self.weak_per_batch)
            ] if self.weak_per_batch else []
            remaining = self.batch_size - len(strong) - len(weak)
            ordinary = [
                int(all_indices[(batch_number * remaining + offset) % len(all_indices)])
                for offset in range(remaining)
            ]
            batch = strong + weak + ordinary
            rng.shuffle(batch)
            yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sensory/molecules.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_sensory"))
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--val-fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--paired-per-batch", type=int, default=2)
    parser.add_argument("--weak-paired-per-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--weak-temperature", type=float, default=0.2)
    parser.add_argument("--weak-taste-weight", type=float, default=0.15)
    parser.add_argument("--weak-contrastive-weight", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-mixtures", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def macro_f1(logits: torch.Tensor, targets: torch.Tensor, labels: tuple[str, ...]) -> dict[str, float]:
    probabilities = torch.sigmoid(logits).cpu().numpy()
    expected = targets.cpu().numpy()
    scores: dict[str, float] = {}
    for column, label in enumerate(labels):
        observed = expected[:, column] >= 0
        scores[label] = float(
            f1_score(
                expected[observed, column],
                (probabilities[observed, column] >= 0.5).astype(int),
                zero_division=0,
            )
        ) if observed.any() else float("nan")
    scores["macro"] = float(np.nanmean(list(scores.values())))
    return scores


@torch.no_grad()
def evaluate(model: CrossSensoryModel, loader: DataLoader, device: torch.device, amp_enabled: bool) -> dict[str, object]:
    model.eval()
    odor_logits, odor_targets, taste_logits, taste_targets = [], [], [], []
    for batch_inputs, odor, taste, _, _, _ in loader:
        batch_inputs = move_to_device(batch_inputs, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(batch_inputs)
        odor_logits.append(outputs["odor_logits"].float().cpu())
        taste_logits.append(outputs["taste_logits"].float().cpu())
        odor_targets.append(odor)
        taste_targets.append(taste)
    odor_logits = torch.cat(odor_logits)
    odor_targets = torch.cat(odor_targets)
    taste_logits = torch.cat(taste_logits)
    taste_targets = torch.cat(taste_targets)
    odor = macro_f1(odor_logits, odor_targets, tuple(ODOR_FAMILIES))
    taste = macro_f1(taste_logits, taste_targets, TASTE_LABELS)
    return {"odor": odor, "taste": taste, "score": (odor["macro"] + taste["macro"]) / 2}


def make_collate(backbone: torch.nn.Module):
    def collate(samples):
        features, odor, taste, weak_taste, strong_paired, weak_only_paired = zip(*samples)
        # UniMol's native collator expects (feature, label) pairs.  Labels are
        # collated separately because they contain masked multi-task targets.
        batch_inputs, _ = backbone.batch_collate_fn([(feature, 0) for feature in features])
        return (
            batch_inputs, torch.stack(odor), torch.stack(taste), torch.stack(weak_taste),
            torch.stack(strong_paired), torch.stack(weak_only_paired),
        )
    return collate


def main() -> None:
    args = parse_args()
    if args.test_fold == args.val_fold:
        raise ValueError("test-fold and val-fold must differ")
    set_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This Colab workflow requires a CUDA runtime. Enable a GPU and rerun.")

    frame = pd.read_parquet(args.data)
    schema_versions = set(frame.get("schema_version", pd.Series(dtype="string")).dropna().astype(str))
    if schema_versions != {SCHEMA_VERSION}:
        observed = ", ".join(sorted(schema_versions)) or "missing"
        raise RuntimeError(
            f"Expected {SCHEMA_VERSION} data, found {observed}. "
            "Rerun `python -m src.dataset.sensory ...` before training."
        )
    keep = frame["odor_known"] | frame["taste_known"]
    if not args.include_mixtures:
        keep &= ~frame["is_mixture"]
    frame = frame.loc[keep].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No trainable molecules after filtering.")
    train_mask = ~frame["fold"].isin([args.test_fold, args.val_fold]).to_numpy()
    val_mask = frame["fold"].eq(args.val_fold).to_numpy()
    test_mask = frame["fold"].eq(args.test_fold).to_numpy()
    if int(frame.loc[train_mask, "paired"].sum()) < 2:
        raise RuntimeError("Training split has fewer than two exact odor--taste pairs.")

    from unimol_tools.data import DataHub
    from unimol_tools.models import UniMolModel

    print(f"device={device}; molecules={len(frame)}; train/val/test={train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    print(f"exact strong pairs: train={int(frame.loc[train_mask, 'paired'].sum())}, val={int(frame.loc[val_mask, 'paired'].sum())}, test={int(frame.loc[test_mask, 'paired'].sum())}")
    print(f"weak-only pairs: train={int((frame.loc[train_mask, 'weak_paired'] & ~frame.loc[train_mask, 'paired']).sum())}, val={int((frame.loc[val_mask, 'weak_paired'] & ~frame.loc[val_mask, 'paired']).sum())}, test={int((frame.loc[test_mask, 'weak_paired'] & ~frame.loc[test_mask, 'paired']).sum())}")
    hub = DataHub(
        data=frame["canonical_smiles"].tolist(), task="repr", is_train=False,
        data_type="molecule", model_name="unimolv1", batch_size=4,
        remove_hs=False, use_cuda=True, use_ddp=False, use_gpu="0", multi_process=False,
    )
    features = hub.data["unimol_input"]
    odor_targets = build_masked_targets(frame, tuple(ODOR_FAMILIES), "odor_labels", "odor_known")
    # TASTE_LABELS is the pre-declared, adequately supported three-label task.
    # Sour and salty remain in the parquet for separate low-shot probes.
    taste_targets = build_masked_targets(frame, TASTE_LABELS, "taste_strong_labels", "taste_known")
    weak_taste_targets = build_masked_targets(
        frame, TASTE_LABELS, "taste_weak_labels", "taste_weak_known"
    )
    strong_paired = frame["paired"].to_numpy(dtype=bool)
    weak_only_paired = (frame["weak_paired"] & ~frame["paired"]).to_numpy(dtype=bool)

    backbone = UniMolModel(output_dim=1, data_type="molecule", remove_hs=False).to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    attention_names = [
        name for name, module in backbone.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith("in_proj")
    ]
    target_names = attention_names[-args.lora_layers:]
    if not target_names or apply_lora(backbone, r=args.lora_rank, alpha=2 * args.lora_rank, target_names=target_names) == 0:
        raise RuntimeError("Uni-Mol attention projections were not found for LoRA.")
    model = CrossSensoryModel(
        backbone=backbone.to(device), embedding_dim=512, odor_dim=len(ODOR_FAMILIES),
        taste_dim=len(TASTE_LABELS), projection_dim=args.projection_dim,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"LoRA targets: {target_names}; trainable parameters: {sum(p.numel() for p in trainable):,}")

    collate = make_collate(model.backbone)
    train_dataset = SensoryDataset(
        [features[index] for index in np.flatnonzero(train_mask)], odor_targets[train_mask], taste_targets[train_mask],
        weak_taste_targets[train_mask], strong_paired[train_mask], weak_only_paired[train_mask],
    )
    val_dataset = SensoryDataset(
        [features[index] for index in np.flatnonzero(val_mask)], odor_targets[val_mask], taste_targets[val_mask],
        weak_taste_targets[val_mask], strong_paired[val_mask], weak_only_paired[val_mask],
    )
    test_dataset = SensoryDataset(
        [features[index] for index in np.flatnonzero(test_mask)], odor_targets[test_mask], taste_targets[test_mask],
        weak_taste_targets[test_mask], strong_paired[test_mask], weak_only_paired[test_mask],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=MultiPairBatchSampler(
            strong_paired[train_mask], weak_only_paired[train_mask], args.batch_size,
            args.paired_per_batch, args.weak_paired_per_batch, args.seed,
        ),
        num_workers=0, pin_memory=True, collate_fn=collate,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"fold{args.test_fold}_best.pt"
    start_epoch, best_score, stale_epochs = 0, float("-inf"), 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["validation"]["score"])
        print(f"resuming at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = {name: 0.0 for name in ("total", "odor", "taste", "strong_contrastive", "weak_taste", "weak_contrastive")}
        for batch_inputs, odor, taste, weak_taste, batch_strong, batch_weak in train_loader:
            batch_inputs = move_to_device(batch_inputs, device)
            odor, taste, weak_taste = odor.to(device), taste.to(device), weak_taste.to(device)
            batch_strong, batch_weak = batch_strong.to(device), batch_weak.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                outputs = model(batch_inputs)
                terms = cross_sensory_loss(
                    outputs, odor, taste, batch_strong,
                    contrastive_weight=args.contrastive_weight,
                    temperature=args.temperature,
                    weak_taste_targets=weak_taste,
                    weak_paired_mask=batch_weak,
                    weak_taste_weight=args.weak_taste_weight,
                    weak_contrastive_weight=args.weak_contrastive_weight,
                    weak_temperature=args.weak_temperature,
                )
            scaler.scale(terms["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            for name, value in terms.items():
                losses[name] += float(value.detach())
        validation = evaluate(model, val_loader, device, amp_enabled=True)
        averaged = {name: value / len(train_loader) for name, value in losses.items()}
        print(
            f"epoch {epoch + 1:02d} loss={averaged['total']:.4f} "
            f"odor_f1={validation['odor']['macro']:.4f} taste_core_f1={validation['taste']['macro']:.4f} "
            f"strong_nce={averaged['strong_contrastive']:.4f} "
            f"weak_bce={averaged['weak_taste']:.4f} weak_nce={averaged['weak_contrastive']:.4f} "
            f"val={validation['score']:.4f}"
        )
        if validation["score"] > best_score:
            best_score = validation["score"]
            stale_epochs = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "validation": validation, "target_names": target_names, "odor_labels": list(ODOR_FAMILIES),
                "taste_labels": list(TASTE_LABELS), "test_fold": args.test_fold, "val_fold": args.val_fold,
                "weak_guidance": {
                    "weak_taste_weight": args.weak_taste_weight,
                    "weak_contrastive_weight": args.weak_contrastive_weight,
                    "weak_temperature": args.weak_temperature,
                },
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {args.patience} stale validation epochs")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, test_loader, device, amp_enabled=True)
    result = {
        "validation": checkpoint["validation"],
        "test": test,
        "task_definition": {
            "core_taste_labels": list(TASTE_LABELS),
            "low_shot_taste_labels": list(LOW_SHOT_TASTE_LABELS),
        },
        "weak_guidance": checkpoint["weak_guidance"],
        "checkpoint": str(checkpoint_path),
    }
    (args.output_dir / f"fold{args.test_fold}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
