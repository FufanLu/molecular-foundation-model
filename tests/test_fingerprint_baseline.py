"""End-to-end test of the fingerprint baseline on a tiny synthetic frame."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fingerprint_baseline import (
    decision_scores,
    fit_label_models,
    morgan_features,
    run_fold,
)

RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

SMILES = [
    "CCO", "CC(=O)O", "c1ccccc1", "CC(C)C", "CCCCCCCC", "OC(=O)c1ccccc1",
    "CC(C)(C)O", "c1ccncc1", "CCC=O", "CCS", "C1CCCCC1", "CC(=O)C",
]


def synthetic_frame(n_rows: int = 48) -> pd.DataFrame:
    rows = []
    for index in range(n_rows):
        smile = SMILES[index % len(SMILES)]
        rows.append({
            "canonical_smiles": f"{smile}.[Na]" if index % 11 == 0 else smile,
            "is_mixture": index % 11 == 0,
            "fold": index % 5,
            "schema_version": "sensory-v3",
            "odor_labels": ["fruity"] if index % 2 == 0 else ["woody", "green"],
            "odor_known": True,
            "taste_strong_labels": ["sweet"] if index % 3 else ["bitter"],
            "taste_known": index % 4 != 0,
            "taste_weak_labels": [],
            "taste_weak_known": False,
        })
    return pd.DataFrame(rows)


class _Args:
    regularisation = 1.0
    max_iter = 200
    seed = 42
    radius = 2
    n_bits = 256


@unittest.skipUnless(RDKIT_AVAILABLE, "rdkit is not installed")
class FingerprintBaselineTest(unittest.TestCase):
    def test_end_to_end_one_fold(self) -> None:
        from src.dataset.sensory import ODOR_FAMILIES, TASTE_LABELS, build_masked_targets

        frame = synthetic_frame()
        frame = frame.loc[(frame["odor_known"] | frame["taste_known"]) & ~frame["is_mixture"]].reset_index(drop=True)
        features = morgan_features(frame["canonical_smiles"].tolist(), radius=2, n_bits=256)
        self.assertEqual(features.shape, (len(frame), 256))

        odor_targets = build_masked_targets(frame, tuple(ODOR_FAMILIES), "odor_labels", "odor_known")
        taste_targets = build_masked_targets(frame, TASTE_LABELS, "taste_strong_labels", "taste_known")

        result = run_fold(_Args(), features, frame, odor_targets, taste_targets, test_fold=0, val_fold=1)
        for block in ("validation", "test", "test_at_fixed_threshold"):
            self.assertIn(block, result)
            self.assertTrue(np.isfinite(result[block]["odor"]["macro"]))
            self.assertTrue(np.isfinite(result[block]["taste"]["macro"]))
            self.assertTrue(np.isfinite(result[block]["score"]))
        self.assertEqual(result["split"], {"test_fold": 0, "val_fold": 1})
        self.assertEqual(len(result["thresholds"]["odor"]), len(ODOR_FAMILIES))
        self.assertEqual(len(result["thresholds"]["taste"]), len(TASTE_LABELS))
        # Aggregator contract: these fields must exist and be JSON-serialisable.
        json.dumps(result)
        self.assertIn("task_definition", result)
        self.assertIn("alignment", result)

    def test_single_class_label_gets_sentinel_scores(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.random((20, 32), dtype=np.float32)
        targets = np.ones((20, 1))  # no negatives observed anywhere
        models = fit_label_models(features, targets, regularisation=1.0, max_iter=100, seed=42)
        self.assertEqual(models, [None])
        scores = decision_scores(models, features)
        self.assertTrue((scores < -9.0).all())


if __name__ == "__main__":
    unittest.main()
