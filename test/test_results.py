from navigation_model.analysis.results import MultiResult

import unittest
import numpy as np
import numpy.testing as npt


class TestMultiResult(unittest.TestCase):

    def test_array(self):
        res1 = {
            "a": np.zeros(10),
        }
        res2 = {
            "a": np.ones(10),
        }
        mr = MultiResult(res1, res2)
        npt.assert_allclose(mr["a"].data(), [np.zeros(10), np.ones(10)])
        npt.assert_allclose(mr["a"].mean(), 0.5 * np.ones(10))
        npt.assert_allclose(mr["a"].median(), 0.5 * np.ones(10))
        npt.assert_allclose(mr["a"].std(), 0.5 * np.ones(10))
        npt.assert_allclose(mr["a"].percentile(), (0.25 * np.ones(10), 0.75 * np.ones(10)))

    def test_matrix(self):
        res1 = {
            "a": np.zeros((10, 10)),
        }
        res2 = {
            "a": np.ones((10, 10)),
        }
        mr = MultiResult(res1, res2)
        npt.assert_allclose(mr["a"].data(), [np.zeros((10, 10)), np.ones((10, 10))])
        npt.assert_allclose(mr["a"].mean(), 0.5 * np.ones((10, 10)))
        npt.assert_allclose(mr["a"].median(), 0.5 * np.ones((10, 10)))
        npt.assert_allclose(mr["a"].std(), 0.5 * np.ones((10, 10)))
        npt.assert_allclose(mr["a"].percentile(), (0.25 * np.ones((10, 10)), 0.75 * np.ones((10, 10))))

    def test_float(self):
        res1 = {
            "a": 0.0,
        }
        res2 = {
            "a": 1.0,
        }
        mr = MultiResult(res1, res2)
        npt.assert_allclose(mr["a"].data(), [0.0, 1.0])
        npt.assert_allclose(mr["a"].mean(), 0.5)
        npt.assert_allclose(mr["a"].median(), 0.5)
        npt.assert_allclose(mr["a"].std(), 0.5)
        npt.assert_allclose(mr["a"].percentile(), (0.25, 0.75))

    def test_dict(self):
        res1 = {
            "a": {"b": 0.0}
        }
        res2 = {
            "a": {"b": 1.0}
        }
        mr = MultiResult(res1, res2)
        npt.assert_allclose(mr["a"]["b"].data(), [0.0, 1.0])
        npt.assert_allclose(mr["a"]["b"].mean(), 0.5)
        npt.assert_allclose(mr["a"]["b"].median(), 0.5)
        npt.assert_allclose(mr["a"]["b"].std(), 0.5)
        npt.assert_allclose(mr["a"]["b"].percentile(), (0.25, 0.75))

    def test_more_results(self):
        res1 = {
            "a": 0.0,
        }
        res2 = {
            "a": 1.5,
        }
        res3 = {
            "a": 2.0,
        }
        mr = MultiResult(res1, res2)
        mr.append(res3)
        npt.assert_allclose(mr["a"].data(), [0.0, 1.5, 2.0])
        npt.assert_allclose(mr["a"].mean(), np.mean([0.0, 1.5, 2.0]))
        npt.assert_allclose(mr["a"].median(), 1.5)
        npt.assert_allclose(mr["a"].std(), np.std([0.0, 1.5, 2.0]))
        npt.assert_allclose(mr["a"].percentile(), (0.75, 1.75))
