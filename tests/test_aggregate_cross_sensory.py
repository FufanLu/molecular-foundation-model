import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_cross_sensory.py"


def metrics(fold: int, score: float) -> dict:
    return {
        "split": {"test_fold": fold, "val_fold": (fold + 1) % 5},
        "task_definition": {
            "core_taste_labels": ["sweet", "bitter", "umami"],
            "low_shot_taste_labels": ["sour", "salty"],
        },
        "alignment": {
            "prototype_weight": 0.05,
            "strong_alignment_weight": 0.05,
            "strong_temperature": 0.07,
            "weak_taste_weight": 0.02,
            "weak_contrastive_weight": 0.01,
            "weak_temperature": 0.5,
        },
        "test": {
            "odor": {"fruity": score, "macro": score},
            "taste": {"sweet": score, "macro": score},
            "score": score,
        },
    }


class AggregateCrossSensoryTest(unittest.TestCase):
    def test_writes_fold_mean_and_standard_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for fold, score in ((0, 0.4), (1, 0.6)):
                path = root / f"fold{fold}_metrics.json"
                path.write_text(json.dumps(metrics(fold, score)), encoding="utf-8")
                paths.append(path)
            output = root / "summary"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--metrics", *(str(path) for path in paths), "--output-dir", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["folds"], [0, 1])
            self.assertEqual(summary["n_folds"], 2)
            self.assertAlmostEqual(summary["test"]["score"]["mean"], 0.5)
            self.assertAlmostEqual(summary["test"]["score"]["std"], 2 ** -0.5 / 5)
            self.assertIn("Core taste labels: sweet, bitter, umami", (output / "summary.md").read_text())

    def test_rejects_mixed_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "fold0_metrics.json", root / "fold1_metrics.json"
            first.write_text(json.dumps(metrics(0, 0.4)), encoding="utf-8")
            incompatible = metrics(1, 0.6)
            incompatible["alignment"]["weak_temperature"] = 0.2
            second.write_text(json.dumps(incompatible), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--metrics", str(first), str(second), "--output-dir", str(root / "summary")],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different alignment", result.stderr)


if __name__ == "__main__":
    unittest.main()
