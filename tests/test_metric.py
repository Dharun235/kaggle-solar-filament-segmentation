import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.postprocess import pq_score


class OfficialPQTests(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((2048, 2048), dtype=bool)
        self.a[100:200, 100:200] = True

    def test_perfect_match(self):
        self.assertEqual(pq_score([self.a], [self.a]), 1.0)

    def test_strict_half_iou_is_not_match(self):
        b = self.a.copy()
        b[100:150, 100:200] = False
        self.assertAlmostEqual(pq_score([self.a], [b]), 0.0)

    def test_empty_both_is_perfect(self):
        self.assertEqual(pq_score([], []), 1.0)


if __name__ == "__main__":
    unittest.main()
