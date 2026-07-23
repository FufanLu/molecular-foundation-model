"""Morgan fingerprint + per-label logistic regression baseline (2D control).

This is the 2D control for the 3D Uni-Mol LoRA runs.  It uses identical
scaffold folds, identical masked targets, and the identical validation-locked
threshold protocol, and writes aggregator-compatible ``fold*_metrics.json``
files.  It needs no GPU and no Uni-Mol featurisation, so it answers "does the
3D foundation model beat a free fingerprint?" on exactly the same evidence.

Typical Colab usage after dataset preparation:

    PYTHONPATH=. python scripts/fingerprint_baseline.py \
      --data data/processed/sensory/molecules.parquet \
      --output-dir outputs/v3_fingerprint_lr --folds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.dataset.sensory import (
    LOW_SHOT_TASTE_LABELS,
    ODOR_FAMILIES,
    SCHEMA_VERSION,
    TASTE_LABELS,
    build_masked_targets,
)
from src.sensory.metrics import macro_f1, select_thresholds

NEGATIVE_SENTINEL_LOGIT = -10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sensory/molecules.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v3_fingerprint_lr"))
    parser.add_argument("--folds", type=str, default=None, help="Comma-separated test folds; default is every fold.")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--regularisation", type=float, default=1.0, help="Logistic regression C.")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-mixtures", action="store_true")
    return parser.parse_args()


def morgan_features(smiles: list[str], radius: int, n_bits: int) -> np.ndarray:
    """Dense float32 Morgan fingerprint matrix; invalid SMILES get zero rows."""
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    features = np.zeros((len(smiles), n_bits), dtype=np.float32)
    invalid = 0
    for row, smile in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            invalid += 1
            continue
        bit_vector = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(bit_vector, features[row])
    if invalid:
        print(f"warning: {invalid} SMILES could not be parsed and use zero fingerprints")
    return features


def fit_label_models(
    features: np.ndarray,
    targets: np.ndarray,
    regularisation: float,
    max_iter: int,
    seed: int,
) -> list[object | None]:
    """One balanced logistic regression per label on observed rows only.

    Labels with a single observed class in the training split return ``None``;
    their decision score is a constant negative sentinel so they predict
    negative at any reasonable threshold.
    """
    from sklearn.linear_model import LogisticRegression

    models: list[object | None] = []
    for column in range(targets.shape[1]):
        observed = targets[:, column] >= 0
        classes = np.unique(targets[observed, column])
        if classes.size < 2:
            models.append(None)
            continue
        model = LogisticRegression(
            C=regularisation, solver="liblinear", class_weight="balanced",
            max_iter=max_iter, random_state=seed,
        )
        model.fit(features[observed], targets[observed, column])
        models.append(model)
    return models


def decision_scores(models: list[object | None], features: np.ndarray) -> np.ndarray:
    scores = np.full((features.shape[0], len(models)), NEGATIVE_SENTINEL_LOGIT, dtype=np.float64)
    for column, model in enumerate(models):
        if model is not None:
            scores[:, column] = model.decision_function(features)
    return scores


def scored_report(
    odor_logits: np.ndarray,
    odor_targets: np.ndarray,
    taste_logits: np.ndarray,
    taste_targets: np.ndarray,
    odor_thresholds: np.ndarray | None,
    taste_thresholds: np.ndarray | None,
) -> dict[str, object]:
    """Mirror scripts/train_cross_sensory.py's report shape exactly."""
    odor = macro_f1(odor_logits, odor_targets, tuple(ODOR_FAMILIES), odor_thresholds)
    taste = macro_f1(taste_logits, taste_targets, TASTE_LABELS, taste_thresholds)
    return {"odor": odor, "taste": taste, "score": (odor["macro"] + taste["macro"]) / 2}


def resolve_fold_plan(args: argparse.Namespace, frame: pd.DataFrame) -> list[tuple[int, int]]:
    n_folds = int(frame["fold"].max()) + 1
    if args.folds is None:
        test_folds = list(range(n_folds))
    else:
        test_folds = [int(part.strip()) for part in args.folds.split(",")]
        for test_fold in test_folds:
            if not 0 <= test_fold < n_folds:
                raise ValueError(f"fold {test_fold} is outside the available 0..{n_folds - 1}")
    return [(test_fold, (test_fold + 1) % n_folds) for test_fold in test_folds]


def run_fold(
    args: argparse.Namespace,
    features: np.ndarray,
    frame: pd.DataFrame,
    odor_targets: np.ndarray,
    taste_targets: np.ndarray,
    test_fold: int,
    val_fold: int,
) -> dict[str, object]:
    train_mask = ~frame["fold"].isin([test_fold, val_fold]).to_numpy()
    val_mask = frame["fold"].eq(val_fold).to_numpy()
    test_mask = frame["fold"].eq(test_fold).to_numpy()
    print(f"fold {test_fold} (val {val_fold}): train/val/test={train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    odor_models = fit_label_models(
        features[train_mask], odor_targets[train_mask], args.regularisation, args.max_iter, args.seed)
    taste_models = fit_label_models(
        features[train_mask], taste_targets[train_mask], args.regularisation, args.max_iter, args.seed)

    val_odor_logits = decision_scores(odor_models, features[val_mask])
    val_taste_logits = decision_scores(taste_models, features[val_mask])
    test_odor_logits = decision_scores(odor_models, features[test_mask])
    test_taste_logits = decision_scores(taste_models, features[test_mask])

    # Thresholds are fit on validation only and locked; test is evaluated once.
    odor_thresholds = select_thresholds(val_odor_logits, odor_targets[val_mask])
    taste_thresholds = select_thresholds(val_taste_logits, taste_targets[val_mask])

    validation = scored_report(
        val_odor_logits, odor_targets[val_mask], val_taste_logits, taste_targets[val_mask],
        odor_thresholds, taste_thresholds,
    )
    test = scored_report(
        test_odor_logits, odor_targets[test_mask], test_taste_logits, taste_targets[test_mask],
        odor_thresholds, taste_thresholds,
    )
    test_fixed = scored_report(
        test_odor_logits, odor_targets[test_mask], test_taste_logits, taste_targets[test_mask],
        None, None,
    )
    result = {
        "validation": validation,
        "test": test,
        "test_at_fixed_threshold": test_fixed,
        "thresholds": {"odor": odor_thresholds.tolist(), "taste": taste_thresholds.tolist()},
        "split": {"test_fold": test_fold, "val_fold": val_fold},
        "task_definition": {
            "core_taste_labels": list(TASTE_LABELS),
            "low_shot_taste_labels": list(LOW_SHOT_TASTE_LABELS),
            "threshold_selection": "validation",
            "pos_weight_cap": None,
        },
        "alignment": {
            "model": "morgan_fingerprint_logreg",
            "radius": args.radius,
            "n_bits": args.n_bits,
            "class_weight": "balanced",
            "solver": "liblinear",
            "C": args.regularisation,
        },
    }
    print(
        f"fold {test_fold}: val={validation['score']:.4f} test={test['score']:.4f} "
        f"(odor {test['odor']['macro']:.4f} / taste {test['taste']['macro']:.4f})"
    )
    return result


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.data)
    schema_versions = set(frame.get("schema_version", pd.Series(dtype="string")).dropna().astype(str))
    if schema_versions != {SCHEMA_VERSION}:
        observed = ", ".join(sorted(schema_versions)) or "missing"
        raise RuntimeError(
            f"Expected {SCHEMA_VERSION} data, found {observed}. "
            "Rerun `python -m src.dataset.sensory ...` before the baseline."
        )
    keep = frame["odor_known"] | frame["taste_known"]
    if not args.include_mixtures:
        keep &= ~frame["is_mixture"]
    frame = frame.loc[keep].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No evaluable molecules after filtering.")

    fold_plan = resolve_fold_plan(args, frame)
    print(f"molecules={len(frame)}; fold plan (test, val)={fold_plan}")
    features = morgan_features(frame["canonical_smiles"].tolist(), args.radius, args.n_bits)
    odor_targets = build_masked_targets(frame, tuple(ODOR_FAMILIES), "odor_labels", "odor_known")
    taste_targets = build_masked_targets(frame, TASTE_LABELS, "taste_strong_labels", "taste_known")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for test_fold, val_fold in fold_plan:
        metrics_path = args.output_dir / f"fold{test_fold}_metrics.json"
        if metrics_path.exists():
            print(f"fold {test_fold}: {metrics_path} already exists, skipping")
            continue
        result = run_fold(args, features, frame, odor_targets, taste_targets, test_fold, val_fold)
        metrics_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
