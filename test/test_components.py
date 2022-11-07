import unittest

import numpy as np
import numpy.testing as npt

from scipy.stats import vonmises

from navigation_model.simulation.behavioural_components import AnxietyComponent, BiomechanicalCostComponent, \
    ExperienceComponent, BiomechPersistenceComponent, ValueTableComponent
from navigation_model.simulation.maze import StandardTiles


class TestAnxietyComponent(unittest.TestCase):

    def setUp(self) -> None:
        self.p1 = 1.0
        self.p2 = 0.6
        self.p3 = 0.5
        self.W_anx = 0.8

        self.pos_comp = AnxietyComponent(W_anx=self.W_anx, p1=self.p1, p2=self.p2, p3=self.p3)

    def test_parameters(self):
        self.assertEqual({"W_anx": self.W_anx, "p1": self.p1, "p2": self.p2, "p3": self.p3}, self.pos_comp.parameters)

    def test_update(self):
        next_tiles = [StandardTiles.TileType.CORNER, StandardTiles.TileType.CENTRE, StandardTiles.TileType.WALL,
                      StandardTiles.TileType.VOID, StandardTiles.TileType.CENTRE]
        self.pos_comp.update(next_tiles)
        npt.assert_allclose(np.array([1, 0.3, 0.6, -1, 0.3]), self.pos_comp.values)
        npt.assert_allclose(self.W_anx * self.pos_comp.values, self.pos_comp.weighted_values)

    def test_active(self):
        next_tiles = [StandardTiles.TileType.CORNER]
        self.pos_comp.update(next_tiles)
        self.pos_comp.active = False
        self.assertEqual(self.pos_comp.values, 0)
        self.assertEqual(self.pos_comp.weighted_values, 0)


class TestBiomechanicalCostComponent(unittest.TestCase):

    def setUp(self) -> None:
        self.W_bcost = 0.7
        self.k = 0.5
        self.gamma = 2.0

        possible_actions = [[0, 0], [1, 0], [2, 0], [0, -1]]
        self.bcost_comp = BiomechanicalCostComponent(possible_actions, W_bcost=self.W_bcost, k=self.k, gamma=self.gamma)

    def test_parameters(self):
        self.assertEqual({"W_bcost": self.W_bcost, "k": self.k, "gamma": self.gamma}, self.bcost_comp.parameters)

    def test_update(self):
        is_state_possible = [True, True, True, False]
        v = vonmises(self.k).pdf(0.0)

        self.bcost_comp.update(True, is_state_possible)
        npt.assert_allclose(self.bcost_comp.values, [0.0, v, self.gamma * v])
        npt.assert_allclose(self.W_bcost * self.bcost_comp.values, self.bcost_comp.weighted_values)

        self.bcost_comp.update(False, is_state_possible)
        npt.assert_allclose(self.bcost_comp.values, np.zeros(3))


class TestExperienceComponent(unittest.TestCase):

    def setUp(self) -> None:
        self.maze_shape = (7, 7)
        self.theta_home = 30
        self.theta_explo = 80
        self.kappa_home = 5.0
        self.kappa_explo = 2.0
        self.xi = 0.006
        self.W_exp = 0.7

        self.exp_comp = ExperienceComponent(self.maze_shape, W_exp=self.W_exp,
                                            theta_home=self.theta_home, theta_explo=self.theta_explo,
                                            kappa_home=self.kappa_home, kappa_explo=self.kappa_explo, xi=self.xi)

    def test_parameters(self):
        self.assertEqual({"W_exp": self.W_exp, "theta_home": self.theta_home, "theta_explo": self.theta_explo,
                          "kappa_home": self.kappa_home, "kappa_explo": self.kappa_explo, "xi": self.xi},
                         self.exp_comp.parameters)

    def test_update(self):
        x = 4
        y = 2
        next_positions = [(3, 2), (4, 3), (4, 2)]
        self.exp_comp.update(x, y, next_positions)

        self.assertEqual(1, self.exp_comp._maze_map_exp[x, y])

        self.assertEqual(0, self.exp_comp._maze_map_exp[3, 2])

        self.assertEqual(self.xi, self.exp_comp._familiarity)

        self.assertEqual(ExperienceComponent.State.HOME, self.exp_comp._ment_state)

        val_exp_expected = [0.0, 0.0, 0.9932620530009145]
        npt.assert_allclose(np.array(val_exp_expected), self.exp_comp.values)
        npt.assert_allclose(self.W_exp * self.exp_comp.values, self.exp_comp.weighted_values)


class TestBiomechPersistenceComponent(unittest.TestCase):

    def setUp(self) -> None:
        self.W_bper = 0.7
        self.bp = 2.0
        self.Wm = 5.0
        self.Wnm = 3.0

        possible_actions = [[0, 0], [1, 0], [2, 0], [0, -1]]
        self.bper_comp = BiomechPersistenceComponent(possible_actions, W_bper=self.W_bper,
                                                     bp=self.bp, Wm=self.Wm, Wnm=self.Wnm)

    def test_parameters(self):
        self.assertEqual({"W_bper": self.W_bper, "bp": self.bp, "Wm": self.Wm, "Wnm": self.Wnm},
                         self.bper_comp.parameters)

    def test_update(self):
        is_state_possible = [True, True, True, False]

        self.bper_comp.update(True, is_state_possible)
        npt.assert_allclose(self.bper_comp.values, [self.Wm * self.bp, self.bp, self.bp])
        npt.assert_allclose(self.W_bper * self.bper_comp.values, self.bper_comp.weighted_values)

        self.bper_comp.update(False, is_state_possible)
        npt.assert_allclose(self.bper_comp.values, [self.bp, self.Wnm * self.bp, self.Wnm * self.bp])
        npt.assert_allclose(self.W_bper * self.bper_comp.values, self.bper_comp.weighted_values)


class TestValueTableComponent(unittest.TestCase):

    def setUp(self) -> None:
        self.W_values = 0.7
        self.v_table = np.random.random((5, 5))
        self.v_table = (self.v_table - np.min(self.v_table)) / (np.max(self.v_table) - np.min(self.v_table)) - 1

        self.vtab_comp = ValueTableComponent(self.v_table, W_values=self.W_values)

    def test_parameters(self):
        self.assertEqual({"W_values": self.W_values}, self.vtab_comp.parameters)

    def test_update(self):
        next_positions = [(3, 2), (4, 3), (4, 2)]
        self.vtab_comp.update(next_positions)

        val_expected = [self.v_table[p[0], p[1]] for p in next_positions]
        npt.assert_allclose(np.array(val_expected), self.vtab_comp.values)
        npt.assert_allclose(self.W_values * self.vtab_comp.values, self.vtab_comp.weighted_values)
