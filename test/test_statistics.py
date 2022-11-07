from navigation_model.analysis.statistics import normalized_distance, manhattan_distance, ResultsDistance

import numpy as np
import numpy.testing as npt
import unittest


class TestDistances(unittest.TestCase):

    def test_normalized(self):
        npt.assert_allclose(normalized_distance(np.zeros(5), np.zeros(5)), 0)
        npt.assert_allclose(normalized_distance(np.ones(5), np.ones(5)), 0)

        x1 = np.ones(5)
        x2 = np.zeros(5)
        d = normalized_distance(x1, x2)
        npt.assert_allclose(d, np.linalg.norm(x1 / np.sum(x1)))

    def test_manhattan(self):
        x1 = (1, 0)
        x2 = (0, 1)
        d = manhattan_distance(x1, x2)
        npt.assert_allclose(d, 2)


class TestResultsDistance(unittest.TestCase):

    def test_results_distance(self):
        compare_against = {"k1": 0, "k2": 1}

        dist = ResultsDistance(manhattan_distance, compare_against, "k1", "k2")
        d = dist({"k1": 1, "k2": 0})
        npt.assert_allclose(d, 2)

        print(dist({"k1": 3, "k2": 0}))
        npt.assert_allclose(dist.mean(), 3)
        npt.assert_allclose(dist.median(), 3)
        npt.assert_allclose(dist.std(), 1)
        npt.assert_allclose(dist.percentile(), (2.5, 3.5))
