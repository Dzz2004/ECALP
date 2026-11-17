import unittest

import torch
import torch.nn.functional as F

from utils.iterative_label_propagation import dynamic_graph_expansion


class IncrementalKNNTest(unittest.TestCase):
    def test_image_neighbors_match_full_recomputation(self):
        images = torch.tensor([
            [2.0, 0.7, 0.4], [0.4, 1.8, 1.3], [1.0, 2.2, 0.2],
            [0.7, 0.5, 2.0], [1.4, 0.25, 0.8], [0.6, 1.2, 0.3],
        ])
        texts = torch.tensor([[3.2, 0.5, 0.6], [0.5, 1.4, 1.1], [1.5, 1.9, 0.3]])
        variance = texts.var(dim=0)
        num_classes = len(texts)

        for context_reweight in (False, True):
            for k_image in (0, 2, 8):
                for count in range(1, len(images) + 1):
                    with self.subTest(context_reweight=context_reweight, k_image=k_image, count=count):
                        # Match main.py: append a temporary copy of the newest image.
                        stream = torch.cat((images[:count], images[count - 1:count]))
                        neighbors, scores = dynamic_graph_expansion(
                            stream, texts, None, 2, k_image, 0, context_reweight=context_reweight,
                        )

                        queries = stream * variance if context_reweight else stream
                        dense = F.cosine_similarity(queries[:, None, :], stream[None, :, :], dim=-1)
                        k = min(k_image, len(stream))
                        expected_scores, expected_neighbors = dense.sort(dim=1, descending=True, stable=True)
                        torch.testing.assert_close(scores[num_classes:, :k], expected_scores[:, :k])
                        torch.testing.assert_close(neighbors[num_classes:, :k] - num_classes, expected_neighbors[:, :k])


if __name__ == '__main__':
    unittest.main()
