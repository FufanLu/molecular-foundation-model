import unittest

import numpy as np

from src.sensory.metrics import (
    compute_pos_weight,
    macro_f1,
    profile_retrieval,
    select_thresholds,
)

try:
    import torch
    from src.sensory.losses import masked_bce_with_logits
except ModuleNotFoundError:
    torch = None


class ComputePosWeightTest(unittest.TestCase):
    def test_ratio_ignores_unknown_entries(self):
        targets = np.array([
            [1.0, -1.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        weight = compute_pos_weight(targets)
        # column 0: 2 negatives / 1 positive; column 1: 1 negative / 1 positive
        np.testing.assert_allclose(weight, [2.0, 1.0])

    def test_cap_and_no_positive_fallback(self):
        targets = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        weight = compute_pos_weight(targets, cap=10.0)
        self.assertEqual(weight[0], 2.0)  # 2 negatives / 1 positive
        self.assertEqual(weight[1], 1.0)  # no positives: BCE left unchanged

    def test_never_downweights_positives(self):
        targets = np.array([[1.0], [1.0], [0.0]])
        self.assertEqual(compute_pos_weight(targets)[0], 1.0)

    def test_rejects_invalid_cap(self):
        with self.assertRaises(ValueError):
            compute_pos_weight(np.array([[1.0]]), cap=0.5)


class SelectThresholdsTest(unittest.TestCase):
    def test_lowers_threshold_for_rare_label(self):
        # A rare positive class scored near 0.3: 0.5 misses every positive.
        expected = np.zeros((20, 1))
        expected[[0, 1, 2]] = 1.0
        probabilities = np.full((20, 1), 0.1)
        probabilities[[0, 1, 2]] = 0.3
        logits = np.log(probabilities / (1 - probabilities))
        thresholds = select_thresholds(logits, expected)
        self.assertLess(thresholds[0], 0.5)
        fixed = macro_f1(logits, expected, ("rare",))
        tuned = macro_f1(logits, expected, ("rare",), thresholds)
        self.assertGreater(tuned["rare"], fixed["rare"])

    def test_fallback_without_positives_or_observations(self):
        logits = np.zeros((4, 2))
        targets = np.array([[0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.0, -1.0]])
        thresholds = select_thresholds(logits, targets)
        np.testing.assert_allclose(thresholds, [0.5, 0.5])


class MacroF1Test(unittest.TestCase):
    def test_excludes_unknown_rows_and_reports_nan(self):
        logits = np.array([[2.0, 0.0], [-2.0, 0.0], [5.0, 0.0]])
        targets = np.array([[1.0, -1.0], [0.0, -1.0], [-1.0, -1.0]])
        scores = macro_f1(logits, targets, ("a", "b"))
        self.assertAlmostEqual(scores["a"], 1.0)  # third row is unknown, ignored
        self.assertTrue(np.isnan(scores["b"]))
        self.assertAlmostEqual(scores["macro"], 1.0)

    def test_per_label_thresholds_applied(self):
        # sigmoid(0) = 0.5: threshold 0.5 predicts positive (tp=1, fp=1, F1=2/3),
        # threshold 0.6 predicts negative (tp=0, F1=0) — identical inputs,
        # different outcomes only because thresholds differ per label.
        logits = np.array([[0.0, 0.0], [0.0, 0.0]])
        targets = np.array([[1.0, 1.0], [0.0, 0.0]])
        scores = macro_f1(logits, targets, ("a", "b"), np.array([0.5, 0.6]))
        self.assertAlmostEqual(scores["a"], 2 / 3)
        self.assertAlmostEqual(scores["b"], 0.0)


class ProfileRetrievalTest(unittest.TestCase):
    def test_perfect_ranking_and_self_exclusion(self):
        # Two label profiles; each query's nearest *other* molecule shares its set.
        projections = np.array([
            [1.0, 0.0], [0.99, 0.01],
            [0.0, 1.0], [0.01, 0.99],
        ])
        projections /= np.linalg.norm(projections, axis=1, keepdims=True)
        targets = np.array([
            [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0],
        ])
        result = profile_retrieval(projections, targets, np.array([True, True, True, True]))
        # Self-match is excluded, so recall@1 is earned, not trivial identity.
        self.assertAlmostEqual(result["recall@1"], 1.0)
        self.assertAlmostEqual(result["mrr"], 1.0)
        self.assertEqual(result["queries"], 4)

    def test_unique_label_sets_are_not_counted(self):
        projections = np.eye(3)
        targets = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = profile_retrieval(projections, targets, np.array([True, True, True]))
        self.assertEqual(result["queries"], 0)
        self.assertTrue(np.isnan(result["recall@1"]))

    def test_unobserved_molecules_leave_the_pool(self):
        projections = np.eye(3)
        targets = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, -1.0]])
        result = profile_retrieval(projections, targets, np.array([True, True, False]))
        self.assertEqual(result["queries"], 2)


@unittest.skipIf(torch is None, "PyTorch is available in the molfm/Colab training environment.")
class MaskedBcePosWeightTest(unittest.TestCase):
    def test_pos_weight_scales_positive_loss(self):
        logits = torch.tensor([[0.0, 0.0]])
        targets = torch.tensor([[1.0, 0.0]])
        plain = masked_bce_with_logits(logits, targets)
        weighted = masked_bce_with_logits(
            logits, targets, pos_weight=torch.tensor([4.0, 1.0])
        )
        expected = (4.0 * torch.nn.functional.binary_cross_entropy_with_logits(
            logits[0, 0], targets[0, 0]
        ) + torch.nn.functional.binary_cross_entropy_with_logits(logits[0, 1], targets[0, 1])) / 2
        self.assertAlmostEqual(float(weighted), float(expected), places=6)
        self.assertGreater(float(weighted), float(plain))

    def test_unknown_entries_stay_masked_with_pos_weight(self):
        logits = torch.tensor([[5.0, 0.0]])
        targets = torch.tensor([[-1.0, 0.0]])
        weighted = masked_bce_with_logits(
            logits, targets, pos_weight=torch.tensor([100.0, 1.0])
        )
        reference = masked_bce_with_logits(logits, targets)
        # The unknown entry with logit 5.0 must not contribute at all.
        self.assertAlmostEqual(float(weighted), float(reference), places=6)


if __name__ == "__main__":
    unittest.main()
