import unittest

import numpy as np

from navigation_model.simulation.exploration_model import BehaviouralModel, RandomModel
from navigation_model.simulation.behavioural_components import Component, weight


class TestBehaviouralModel(unittest.TestCase):

    @weight("W_choice")
    class ChoiceComponent(Component):

        _name = "choice"

        def __init__(self, n_next, **kwargs):
            super().__init__(**kwargs)

            self._n_next = n_next

        def update(self, chosen_next, **kwargs):
            self._val = np.zeros(self._n_next) - np.inf
            self._val[chosen_next] = 1

    @weight("W_fixed")
    class FixedComponent(Component):

        _name = "fixed"

        def __init__(self, vals, **kwargs):
            super().__init__(**kwargs)

            self._val = vals

        def update(self, **kwargs):
            pass

    def test_exploration(self):
        possible_positions = np.random.randint(100, size=10)
        cc = TestBehaviouralModel.ChoiceComponent(n_next=len(possible_positions))
        model = BehaviouralModel([cc], (0, 0), 1.0)

        for _ in range(100):
            c = np.random.randint(len(possible_positions))
            n = model.decision_making(possible_positions, chosen_next=c)
            self.assertEqual(n, possible_positions[c])

    def test_same_seed(self):

        possible_positions = np.random.randint(100, size=10)
        possible_positions_cont = np.random.random(len(possible_positions))
        vals = np.random.random(len(possible_positions))
        fc1 = TestBehaviouralModel.FixedComponent(vals=vals)
        fc2 = TestBehaviouralModel.FixedComponent(vals=vals)
        bm1 = BehaviouralModel([fc1], (0, 0), 1.0, seed=12345)
        bm2 = BehaviouralModel([fc2], (0, 0), 1.0, seed=12345)

        for _ in range(100):
            p1 = bm1.decision_making(next_positions_cont=possible_positions_cont)
            p2 = bm2.decision_making(next_positions_cont=possible_positions_cont)
            self.assertEqual(p1, p2)


class TestRandomModel(unittest.TestCase):

    def test_exploration(self):

        rm = RandomModel()
        next_positions_cont = np.random.random(10)

        for _ in range(100):
            p = rm.decision_making(next_positions_cont=next_positions_cont)
            self.assertLess(p, 10)

    def test_same_seed(self):

        rm1 = RandomModel(seed=12345)
        rm2 = RandomModel(seed=12345)
        next_positions_cont = np.random.random(10)

        for _ in range(100):
            p1 = rm1.decision_making(next_positions_cont=next_positions_cont)
            p2 = rm2.decision_making(next_positions_cont=next_positions_cont)
            self.assertEqual(p1, p2)
