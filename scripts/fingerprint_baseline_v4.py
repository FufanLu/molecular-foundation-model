"""Versioned Morgan-fingerprint baseline and ontology-validation controls.

This runner intentionally leaves ``fingerprint_baseline.py`` and its v3
artefacts untouched.  It reuses that module's modelling and metric helpers,
but targets the v4 ontology and adds two explicitly distinct controls:

* ``--shuffle-ontology`` is an ontology permutation.  It preserves
  molecule-to-raw-descriptor evidence and is therefore not a chance baseline.
* ``--shuffle-odor-train-labels`` is a chemistry-null.  It permutes complete
  odor target rows in each training fold while validation and test labels stay
  real, preserving label prevalence, unknown masks, and co-occurrence.
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

from scripts import fingerprint_baseline as v3
from src.dataset.sensory import (
    LOW_SHOT_TASTE_LABELS,
    ODOR_FAMILIES,
    SCHEMA_VERSION,
    TASTE_LABELS,
    _map_terms,
    build_masked_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sensory/molecules.parquet"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", type=str, default=None, help="Comma-separated test folds; default is every fold.")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--regularisation", type=float, default=1.0, help="Logistic-regression C.")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-mixtures", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run folds even when metrics JSONs already exist.")
    parser.add_argument("--shuffle-ontology", type=int, default=None, metavar="SEED",
                        help="Ontology permutation of descriptor-to-family membership; not a chance baseline.")
    parser.add_argument("--shuffle-odor-train-labels", type=int, default=None, metavar="SEED",
                        help="Chemistry-null: permute complete odor target rows in each training split only.")
    parser.add_argument("--source-records", type=Path, default=Path("data/processed/sensory/source_records.parquet"),
                        help="Raw descriptor evidence; required for --shuffle-ontology.")
    return parser.parse_args()


def shuffled_taxonomy(taxonomy: dict[str, set[str]], seed: int) -> dict[str, set[str]]:
    """Permute a strict descriptor partition, preserving family sizes exactly."""
    owner: dict[str, str] = {}
    for family, aliases in taxonomy.items():
        for term in aliases:
            if term in owner:
                raise ValueError(
                    "Ontology permutation requires disjoint descriptor families; "
                    f"{term!r} appears in both {owner[term]!r} and {family!r}."
                )
            owner[term] = family
    return v3.shuffled_taxonomy(taxonomy, seed)


def shuffled_target_rows(targets: np.ndarray, seed: int) -> np.ndarray:
    """Break chemistry-target links while retaining each full label vector."""
    targets = np.asarray(targets)
    return targets[np.random.default_rng(seed).permutation(len(targets))].copy()


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

    train_odor_targets = odor_targets[train_mask]
    if args.shuffle_odor_train_labels is not None:
        # One reproducible but distinct permutation per test fold.
        train_odor_targets = shuffled_target_rows(train_odor_targets, args.shuffle_odor_train_labels + test_fold)
    odor_models = v3.fit_label_models(
        features[train_mask], train_odor_targets, args.regularisation, args.max_iter, args.seed)
    taste_models = v3.fit_label_models(
        features[train_mask], taste_targets[train_mask], args.regularisation, args.max_iter, args.seed)

    val_odor_logits = v3.decision_scores(odor_models, features[val_mask])
    val_taste_logits = v3.decision_scores(taste_models, features[val_mask])
    test_odor_logits = v3.decision_scores(odor_models, features[test_mask])
    test_taste_logits = v3.decision_scores(taste_models, features[test_mask])
    odor_thresholds = v3.select_thresholds(val_odor_logits, odor_targets[val_mask])
    taste_thresholds = v3.select_thresholds(val_taste_logits, taste_targets[val_mask])

    validation = v3.scored_report(
        val_odor_logits, odor_targets[val_mask], val_taste_logits, taste_targets[val_mask],
        odor_thresholds, taste_thresholds,
    )
    test = v3.scored_report(
        test_odor_logits, odor_targets[test_mask], test_taste_logits, taste_targets[test_mask],
        odor_thresholds, taste_thresholds,
    )
    test_fixed = v3.scored_report(
        test_odor_logits, odor_targets[test_mask], test_taste_logits, taste_targets[test_mask], None, None,
    )
    control = (
        "ontology_permutation" if args.shuffle_ontology is not None
        else "training_target_permutation" if args.shuffle_odor_train_labels is not None
        else "real_ontology"
    )
    result = {
        "validation": validation,
        "test": test,
        "test_at_fixed_threshold": test_fixed,
        "thresholds": {"odor": odor_thresholds.tolist(), "taste": taste_thresholds.tolist()},
        "split": {"test_fold": test_fold, "val_fold": val_fold},
        "task_definition": {
            "schema_version": SCHEMA_VERSION,
            "core_taste_labels": list(TASTE_LABELS),
            "low_shot_taste_labels": list(LOW_SHOT_TASTE_LABELS),
            "threshold_selection": "validation",
            "odor_control_protocol": "validation_and_test_targets_remain_real",
        },
        "alignment": {
            "model": "morgan_fingerprint_logreg",
            "radius": args.radius,
            "n_bits": args.n_bits,
            "class_weight": "balanced",
            "solver": "liblinear",
            "C": args.regularisation,
            "odor_control": control,
            "shuffle_ontology_seed": args.shuffle_ontology,
            "shuffle_odor_train_labels_seed": args.shuffle_odor_train_labels,
        },
    }
    print(
        f"fold {test_fold}: val={validation['score']:.4f} test={test['score']:.4f} "
        f"(odor {test['odor']['macro']:.4f} / taste {test['taste']['macro']:.4f})"
    )
    return result


def main() -> None:
    args = parse_args()
    if args.shuffle_ontology is not None and args.shuffle_odor_train_labels is not None:
        raise ValueError("Choose one odor control at a time.")
    frame = pd.read_parquet(args.data)
    schema_versions = set(frame.get("schema_version", pd.Series(dtype="string")).dropna().astype(str))
    if schema_versions != {SCHEMA_VERSION}:
        observed = ", ".join(sorted(schema_versions)) or "missing"
        raise RuntimeError(f"Expected {SCHEMA_VERSION} data, found {observed}. Regenerate the v4 dataset first.")
    keep = frame["odor_known"] | frame["taste_known"]
    if not args.include_mixtures:
        keep &= ~frame["is_mixture"]
    frame = frame.loc[keep].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No evaluable molecules after filtering.")
    real_odor_known_count = int(frame["odor_known"].sum())

    if args.shuffle_ontology is not None:
        if not args.source_records.exists():
            raise RuntimeError(f"--shuffle-ontology needs raw descriptor evidence: {args.source_records} not found")
        taxonomy = shuffled_taxonomy(ODOR_FAMILIES, args.shuffle_ontology)
        terms_by_smiles = v3.molecule_odor_terms(pd.read_parquet(args.source_records))
        frame["odor_labels"] = [
            sorted(_map_terms(terms_by_smiles.get(smiles, ()), taxonomy))
            for smiles in frame["canonical_smiles"]
        ]
        frame["odor_known"] = frame["odor_labels"].apply(bool)
        if int(frame["odor_known"].sum()) != real_odor_known_count:
            raise RuntimeError(
                "Ontology permutation changed the number of odor-known molecules. "
                "Refuse to compare controls with a different observed-label mask."
            )
        print(f"ONTOLOGY PERMUTATION: seed={args.shuffle_ontology}; not a chance baseline.")
    if args.shuffle_odor_train_labels is not None:
        print(
            f"CHEMISTRY NULL: training odor target rows permuted (seed={args.shuffle_odor_train_labels}); "
            "validation/test labels remain real."
        )

    if args.output_dir is None:
        suffix = (
            f"_ontology_shuffle_seed{args.shuffle_ontology}" if args.shuffle_ontology is not None
            else f"_odor_train_permutation_seed{args.shuffle_odor_train_labels}"
            if args.shuffle_odor_train_labels is not None else ""
        )
        args.output_dir = Path(f"outputs/v4_fingerprint_lr{suffix}")
    fold_plan = v3.resolve_fold_plan(args, frame)
    print(f"molecules={len(frame)}; fold plan (test, val)={fold_plan}")
    features = v3.morgan_features(frame["canonical_smiles"].tolist(), args.radius, args.n_bits)
    odor_targets = build_masked_targets(frame, tuple(ODOR_FAMILIES), "odor_labels", "odor_known")
    taste_targets = build_masked_targets(frame, TASTE_LABELS, "taste_strong_labels", "taste_known")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for test_fold, val_fold in fold_plan:
        metrics_path = args.output_dir / f"fold{test_fold}_metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"fold {test_fold}: {metrics_path} already exists, skipping")
            continue
        result = run_fold(args, features, frame, odor_targets, taste_targets, test_fold, val_fold)
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
