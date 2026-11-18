"""Run with: python -m unittest discover -s tests -v"""

import unittest
from collections import defaultdict

import torch

from src.dataset import (DEFAULT_MANIFEST, TrashNetDataset, build_train_val_loaders,
                         read_manifest, train_class_weights)
from src.labels import LABELS, LABEL_TO_INDEX
from src.prepare_data import (ROOT, assign_splits, find_duplicate_candidates,
                              inspect_images, validate_manifest)


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_MANIFEST.exists():
            raise unittest.SkipTest("Phase 2 manifest not prepared")
        cls.rows = read_manifest()

    def test_labels_and_splits(self):
        self.assertEqual(len(self.rows), 2521)
        self.assertEqual(set(LABELS), set(LABEL_TO_INDEX))
        for split in ("train", "val", "test"):
            self.assertEqual(set(LABELS), {r["label"] for r in self.rows if r["split"] == split})

    def test_no_exact_or_group_leakage(self):
        for key in ("sha256", "group_id"):
            by_key = defaultdict(set)
            for row in self.rows:
                by_key[row[key]].add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in by_key.values()))

    def test_near_duplicate_candidates_are_grouped(self):
        import csv
        split_by_path = {row["path"]: row["split"] for row in self.rows}
        with (ROOT / "data/reports/duplicate_candidates.csv").open(newline="", encoding="utf-8") as stream:
            candidates = list(csv.DictReader(stream))
        for pair in candidates:
            if pair["kind"] == "near_same_label":
                self.assertEqual(split_by_path[pair["path_a"]], split_by_path[pair["path_b"]])

    def test_manifest_is_reproducible(self):
        records, _ = inspect_images(ROOT / "data/raw/dataset-resized")
        by_sha = defaultdict(set)
        for record in records:
            by_sha[record["sha256"]].add(record["label"])
        usable = [record for record in records if len(by_sha[record["sha256"]]) == 1]
        _, groups = find_duplicate_candidates(usable)
        reconstructed = assign_splits(usable, groups, 42)
        validate_manifest(reconstructed)
        self.assertEqual(self.rows, reconstructed)

    def test_train_val_api_does_not_load_test(self):
        train, val = build_train_val_loaders(batch_size=2)
        self.assertEqual(train.dataset.split, "train")
        self.assertEqual(val.dataset.split, "val")
        self.assertNotIn("test", {row["split"] for row in train.dataset.rows + val.dataset.rows})

    def test_class_weights_are_train_only(self):
        weights = train_class_weights()
        self.assertEqual(tuple(weights.shape), (6,))
        self.assertTrue(torch.all(weights > 0))

    def test_image_shape_eval_determinism_and_train_variation(self):
        train = TrashNetDataset("train")
        val = TrashNetDataset("val")
        first, label = train[0]
        self.assertEqual(tuple(first.shape), (3, 224, 224))
        self.assertIn(label, range(6))
        self.assertTrue(torch.isfinite(first).all())
        eval_one, _ = val[0]
        eval_two, _ = val[0]
        self.assertTrue(torch.equal(eval_one, eval_two))
        samples = [train[0][0] for _ in range(5)]
        self.assertTrue(any(not torch.equal(samples[0], item) for item in samples[1:]))


if __name__ == "__main__":
    unittest.main()
