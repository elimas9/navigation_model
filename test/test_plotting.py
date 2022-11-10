from navigation_model.visualization.plotting import plot_boxes, plot_multiple_boxes

import unittest
import matplotlib.pyplot as plt
import numpy as np


class TestBoxesPlot(unittest.TestCase):

    def test_plot_boxes_dict(self):
        ax = plt.figure().gca()
        plot_boxes(ax, {"cat1": np.random.random(10),
                        "cat2": np.random.random(10),
                        "cat3": np.random.random(10)})

    def test_plot_boxes_list(self):
        ax = plt.figure().gca()
        plot_boxes(ax,
                   data=[np.random.random(10), np.random.random(10), np.random.random(10)],
                   labels=["cat1", "cat2", "cat3"])
        self.assertRaises(RuntimeError, plot_boxes,
                          ax, [np.random.random(10), np.random.random(10), np.random.random(10)], ["cat1"])

    def test_plot_boxes_single_values_dict(self):
        ax = plt.figure().gca()
        plot_boxes(ax, {"cat1": 1.0, "cat2": 2.0, "cat3": 3.0})

    def test_plot_boxes_single_values_list(self):
        ax = plt.figure().gca()
        plot_boxes(ax, [1.0, 2.0, 3.0], labels=["cat1", "cat2", "cat3"])

    def test_plot_multiple_boxes(self):
        plot_multiple_boxes(plt.figure(),
                            [{"cat1": np.random.random(10),
                              "cat2": np.random.random(10),
                              "cat3": np.random.random(10)},
                             {"cat1": np.random.random(10),
                              "cat2": np.random.random(10),
                              "cat3": np.random.random(10),
                              "cat4": np.random.random(10)},
                             {"cat1": 1.0,
                              "cat2": 2.0,
                              "cat3": 3.0}],
                            rows=2, columns=2,
                            titles=["title1", "title2"])  # should work with less titles

        # we have an error if not enough rows and columns are given
        self.assertRaises(RuntimeError, plot_multiple_boxes,
                          plt.figure, [{}, {}, {}], 2, 1)
