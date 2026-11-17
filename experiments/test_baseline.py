"""Small data-contract and baseline checks; these do not measure real accuracy."""
from pathlib import Path
import tempfile
import unittest

import torch

from run_main_results import clip_accuracy, load_features


class BaselineTests(unittest.TestCase):
    def test_cosine_is_invariant_to_positive_feature_scaling(self):
        texts = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        images = torch.tensor([[0., 7., 0.], [9., 0., 0.], [0., 0., 2.]])
        labels = torch.tensor([1, 0, 2])
        self.assertEqual(clip_accuracy(images, texts, labels), 100.)
        self.assertEqual(clip_accuracy(images, texts * torch.tensor([[8.], [3.], [2.]]), labels), 100.)

    def test_average_views_before_classification(self):
        images = torch.tensor([[[1., 0.], [0.8, 0.2]], [[0., 1.], [0.1, 0.9]]])
        self.assertEqual(clip_accuracy(images, torch.eye(2), torch.tensor([0, 1])), 100.)

    def test_rejects_bad_labels_nonfinite_and_zero_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.pt"
            base = dict(image_features=torch.eye(2), text_features=torch.eye(2),
                        labels=torch.tensor([0, 1]))
            invalid = [{"labels": torch.tensor([0, 2])},
                       {"labels": torch.tensor([0., 1.])},
                       {"labels": torch.tensor([[0], [1]])},
                       {"image_features": torch.tensor([[float("nan"), 0.], [0., 1.]])},
                       {"image_features": torch.tensor([[0., 0.], [0., 1.]])}]
            for change in invalid:
                with self.subTest(change=change):
                    torch.save({**base, **change}, path)
                    with self.assertRaises(ValueError):
                        load_features(path)
            torch.save(base, path)
            _, _, labels, summary = load_features(path)
            self.assertEqual(summary["class_counts"], [1, 1])
            self.assertEqual(len(summary["sha256"]), 64)
            self.assertTrue(torch.equal(labels, base["labels"]))


if __name__ == "__main__":
    unittest.main()
