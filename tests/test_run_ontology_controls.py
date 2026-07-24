from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.run_ontology_controls import build_report, parse_seed_spec


def summary(mean: float, values: list[float]) -> dict:
    return {"test": {"odor": {"macro": {"mean": mean, "values": values}}}}


class OntologyControlRunnerTest(unittest.TestCase):
    def test_parses_seed_ranges(self) -> None:
        self.assertEqual(parse_seed_spec("0:2,7"), [0, 1, 2, 7])
        with self.assertRaises(ValueError):
            parse_seed_spec("2:0")
        with self.assertRaises(ValueError):
            parse_seed_spec("1,1")

    def test_report_records_empirical_tail_and_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.json"
            ontology_0, ontology_1 = root / "ontology_0.json", root / "ontology_1.json"
            chemistry_0, chemistry_1 = root / "chemistry_0.json", root / "chemistry_1.json"
            real.write_text(json.dumps(summary(0.5, [0.4, 0.6])), encoding="utf-8")
            ontology_0.write_text(json.dumps(summary(0.3, [0.2, 0.4])), encoding="utf-8")
            ontology_1.write_text(json.dumps(summary(0.6, [0.5, 0.7])), encoding="utf-8")
            chemistry_0.write_text(json.dumps(summary(0.1, [0.1, 0.1])), encoding="utf-8")
            chemistry_1.write_text(json.dumps(summary(0.2, [0.2, 0.2])), encoding="utf-8")
            args = Namespace(
                data=Path("data.parquet"), source_records=Path("source.parquet"), folds=None,
                radius=2, n_bits=2048, regularisation=1.0, max_iter=1000, seed=42, seeds="0:1",
            )
            report_dir = root / "report"
            build_report(real, {0: ontology_0, 1: ontology_1}, {0: chemistry_0, 1: chemistry_1}, report_dir, args)
            payload = json.loads((report_dir / "control_summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["ontology_permutation"]["upper_tail_p"], 2 / 3)
            self.assertAlmostEqual(payload["ontology_permutation"]["real_percentile"], 0.5)
            self.assertEqual(payload["chemistry_null"]["n_seeds"], 2)
            self.assertIn("Interpretation gate", (report_dir / "control_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
