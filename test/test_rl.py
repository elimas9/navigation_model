import copy
import unittest
import numpy as np

from navigation_model.simulation.rl import PositiveConditioning, NegativeConditioning, TD0, Transition,\
    ShuffledReplays, Update


class TestAlgorithms(unittest.TestCase):

    class MockMouse:
        def move_to(self, _):
            pass

    def test_positive_conditioning(self):

        alpha = np.random.random(1)
        discount_rate = np.random.random(1)

        alg = PositiveConditioning(alpha=alpha, discount_rate=discount_rate, mouse=TestAlgorithms.MockMouse(), maze=None)

        v = np.random.random((3, 3))
        s = (1, 1)
        neighb = [(1, 0), (0, 1), (2, 1), (1, 2)]
        r = 1

        alg.v_table = copy.copy(v)
        alg_update = alg.update(Transition(s=s, r=r, extra={"next_states": neighb}))
        dv = discount_rate * np.max([v[s] for s in neighb])
        dv = alpha * (r + dv - v[s])
        self.assertEqual(alg_update.dv, dv)
        self.assertEqual(alg.v_table[s], v[s] + dv)

    def test_negative_conditioning(self):

        alpha = np.random.random(1)
        discount_rate = np.random.random(1)

        alg = NegativeConditioning(alpha=alpha, discount_rate=discount_rate, mouse=TestAlgorithms.MockMouse(), maze=None)

        v = np.random.random((3, 3))
        s = (1, 1)
        neighb = [(1, 0), (0, 1), (2, 1), (1, 2)]
        r = -1

        alg.v_table = copy.copy(v)
        alg_update = alg.update(Transition(s=s, r=r, extra={"next_states": neighb}))
        dv = discount_rate * np.min([v[s] for s in neighb])
        dv = alpha * (r + dv - v[s])
        self.assertEqual(alg_update.dv, dv)
        self.assertEqual(alg.v_table[s], v[s] + dv)

    def test_td0(self):

        alpha = np.random.random(1)
        discount_rate = np.random.random(1)

        alg = TD0(alpha=alpha, discount_rate=discount_rate)

        v = np.random.random((3, 3))
        s = (1, 1)
        s1 = (np.random.randint(0, 3), np.random.randint(0, 3))
        r = 1

        alg.v_table = copy.copy(v)
        alg_update = alg.update(Transition(s=s, s1=s1, r=r))
        dv = alpha * (r + discount_rate * v[s1] - v[s])
        self.assertEqual(alg_update.dv, dv)
        self.assertEqual(alg.v_table[s], v[s] + dv)


class TestReplayStrategies(unittest.TestCase):

    class CountAlgorithm:

        def __init__(self):
            self._max = 0
            self.counts = []

        def update(self, t):
            if "c" in t.extra:
                self.counts[t.extra["c"]] += 1
            else:
                t.extra["c"] = self._max
                self._max += 1
                self.counts.append(0)
            return Update(transition=t)

    def test_shuffled_replays(self):

        alg = TestReplayStrategies.CountAlgorithm()
        repl = ShuffledReplays(n_times=10)
        n_updates = 200
        for _ in range(n_updates):
            u = alg.update(Transition())
            repl.append(u)

        repl.offline_replays(alg)
        self.assertEqual(sum(alg.counts), n_updates * 10)
        self.assertEqual(max(alg.counts), 10)
