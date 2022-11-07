from navigation_model.optimization.fitness import MultiObjectiveFitness

import numpy as np
import numpy.testing as npt
import unittest

from navigation_model.analysis.statistics import normalized_distance, manhattan_distance, ResultsDistance


class TesttFitness(unittest.TestCase):

    def test_multiobjectivefitness(self):
        compare_against = {"a": 0, "b": 1, "c": np.zeros(5)}

        fitness = MultiObjectiveFitness(compare_against=compare_against)
        fitness.add_objective("obj1", manhattan_distance, "a", "b")
        fitness.add_objective("obj2", normalized_distance, "c")

        d = fitness({"a": 1, "b": 0, "c": np.ones(5)})
        npt.assert_allclose(d["obj1"], 2)
        npt.assert_allclose(d["obj2"], np.linalg.norm(np.ones(5) / 5))

        fitness({"a": 3, "b": 0, "c": np.zeros(5)})
        npt.assert_allclose(fitness.means()["obj1"], 3)
        npt.assert_allclose(fitness.medians()["obj1"], 3)
        npt.assert_allclose(fitness.stds()["obj1"], 1)
        npt.assert_allclose(fitness.percentiles()["obj1"], (2.5, 3.5))
        npt.assert_allclose(fitness.means_list()[0], 3)
        npt.assert_allclose(fitness.medians_list()[0], 3)
        npt.assert_allclose(fitness.stds_list()[0], 1)
        npt.assert_allclose(fitness.percentiles_list()[0], (2.5, 3.5))
