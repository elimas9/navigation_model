import unittest
import numpy as np

from navigation_model.simulation.rl import PositiveConditioning, NegativeConditioning, TD0


class TestAlgorithms(unittest.TestCase):

    def test_positive_conditioning(self):

        alpha = np.random.random(1)
        discount_rate = np.random.random(1)

        alg = PositiveConditioning(alpha=alpha, discount_rate=discount_rate, mouse=None, maze=None)

        v = np.random.random((3, 3))
        s = (1, 1)
        neighb = [(1, 0), (0, 1), (2, 1), (1, 2)]
        r = 1

        alg_update = alg.get_update(v, s, neighb, r)
        dv = discount_rate * np.max([v[s] for s in neighb])
        dv = alpha * (r + dv - v[s])
        self.assertEqual(alg_update, dv)

    def test_negative_conditioning(self):

        alpha = np.random.random(1)
        discount_rate = np.random.random(1)

        alg = NegativeConditioning(alpha=alpha, discount_rate=discount_rate, mouse=None, maze=None)

        v = np.random.random((3, 3))
        s = (1, 1)
        neighb = [(1, 0), (0, 1), (2, 1), (1, 2)]
        r = -1

        alg_update = alg.get_update(v, s, neighb, r)
        dv = discount_rate * np.min([v[s] for s in neighb])
        dv = alpha * (r + dv - v[s])
        self.assertEqual(alg_update, dv)

    def test_td0(self):

        alpha = np.random.random(1)
        discount_rate = np.random.random(1)

        alg = TD0(alpha=alpha, discount_rate=discount_rate)

        v = np.random.random((3, 3))
        s = (1, 1)
        s1 = (np.random.randint(0, 3), np.random.randint(0, 3))
        r = 1

        alg_update = alg.get_update(v, s, s1, r)
        dv = alpha * (r + discount_rate * v[s1] - v[s])
        self.assertEqual(alg_update, dv)
