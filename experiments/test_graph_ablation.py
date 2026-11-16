"""Synthetic checks for the image-image edge ablation."""

import importlib
import unittest

import torch

import utils.iterative_label_propagation as label_propagation


class GraphAblationTest(unittest.TestCase):
    def setUp(self):
        self.lp = importlib.reload(label_propagation)
        self.image_features = torch.tensor(
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]], dtype=torch.float32
        )
        self.text_features = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )

    def graph_inputs(self, include_image_image_edges):
        return self.lp.dynamic_graph_expansion(
            self.image_features,
            self.text_features,
            fewshot_features=None,
            k_text=1,
            k_image=2,
            k_fewshot=0,
            include_image_image_edges=include_image_image_edges,
        )

    def test_disabled_variant_removes_only_image_image_edges(self):
        knn, sim = self.graph_inputs(include_image_image_edges=False)
        num_classes = self.text_features.shape[0]
        image_rows = slice(num_classes, None)
        image_neighbor_mask = knn[image_rows] >= num_classes
        text_neighbor_mask = (knn[image_rows] >= 0) & (knn[image_rows] < num_classes)

        self.assertTrue(torch.all(sim[image_rows][image_neighbor_mask] == 0))
        self.assertTrue(torch.all(sim[image_rows][text_neighbor_mask] != 0))

        adjacency = self.lp.knn_to_weighted_adj_matrix(
            knn, sim, num_nodes=knn.shape[0]
        ).coalesce()
        rows, columns = adjacency.indices()
        image_to_image = (rows >= num_classes) & (columns >= num_classes)
        image_to_text = (rows >= num_classes) & (columns < num_classes)

        self.assertFalse(torch.any(image_to_image))
        self.assertTrue(torch.any(image_to_text))

    def test_default_variant_retains_image_image_edges(self):
        knn, sim = self.graph_inputs(include_image_image_edges=True)
        num_classes = self.text_features.shape[0]
        image_rows = slice(num_classes, None)
        image_neighbor_mask = knn[image_rows] >= num_classes
        self.assertTrue(torch.any(sim[image_rows][image_neighbor_mask] != 0))


if __name__ == "__main__":
    unittest.main()
