import unittest

try:
    import torch
    from src.sensory.losses import (
        molecule_prototype_nce,
        paired_label_set_info_nce,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is available in the molfm/Colab training environment.")
class PrototypeAlignmentTest(unittest.TestCase):
    def test_molecule_prototype_nce_is_finite_and_differentiable(self):
        molecule = torch.nn.functional.normalize(torch.randn(3, 4, requires_grad=True), dim=-1)
        prototypes = torch.nn.functional.normalize(torch.randn(3, 4, requires_grad=True), dim=-1)
        targets = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        loss = molecule_prototype_nce(molecule, prototypes, targets, temperature=0.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_paired_label_set_alignment_needs_two_pairs(self):
        odor = torch.nn.functional.normalize(torch.randn(3, 4, requires_grad=True), dim=-1)
        taste = torch.nn.functional.normalize(torch.randn(2, 4, requires_grad=True), dim=-1)
        odor_targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        taste_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        aligned = paired_label_set_info_nce(
            odor, taste, odor_targets, taste_targets, torch.tensor([True, True]), temperature=0.2
        )
        self.assertTrue(torch.isfinite(aligned))
        aligned.backward()
        one_pair = paired_label_set_info_nce(
            odor.detach(), taste.detach(), odor_targets, taste_targets, torch.tensor([True, False]), temperature=0.2
        )
        self.assertEqual(float(one_pair), 0.0)

    def test_duplicate_label_sets_are_multi_positives(self):
        odor = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
        taste = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
        odor_targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        taste_targets = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        loss = paired_label_set_info_nce(
            odor, taste, odor_targets, taste_targets, torch.tensor([True, True]), temperature=0.2
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
