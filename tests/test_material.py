"""Regression tests for the pair gate and redistribution used by the new variant."""

import unittest

import torch

from src.material_model import GLASS, PLASTIC, agree_to_switch, pair_gate, update_pair


class MaterialTests(unittest.TestCase):
    def test_gate_requires_glass_and_plastic_as_the_two_leading_labels(self):
        logits = torch.zeros(3, 6)
        logits[:, GLASS] = 2
        logits[:, PLASTIC] = 3
        logits[1, 0] = 4
        logits[2, 0] = 1
        self.assertEqual(pair_gate(logits).tolist(), [True, False, True])

    def test_redistribution_can_flip_the_pair_and_preserves_unrelated_probabilities(self):
        logits = torch.zeros(1, 6)
        logits[0, GLASS] = 4
        logits[0, PLASTIC] = 2
        revised = update_pair(logits, torch.tensor([2.0])).softmax(dim=1)
        prior = logits.softmax(dim=1)
        self.assertEqual(revised.argmax(dim=1).item(), PLASTIC)
        self.assertTrue(torch.allclose(revised[:, (GLASS, PLASTIC)].sum(dim=1),
                                       prior[:, (GLASS, PLASTIC)].sum(dim=1)))
        others = [index for index in range(6) if index not in (GLASS, PLASTIC)]
        self.assertTrue(torch.allclose(revised[:, others], prior[:, others]))

    def test_agreement_requires_both_secondary_opinions_to_favor_the_other_label(self):
        base = torch.tensor([-2.0, -2.0, 2.0, -2.0, -2.0])
        specialist = torch.tensor([0.9, 0.9, -0.9, 0.1, -1.0])
        peer = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0])
        self.assertEqual(agree_to_switch(base, specialist, peer, .4).tolist(), [True, False, True, False, False])
