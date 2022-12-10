import unittest
import numpy as np
import numpy.testing as npt

from navigation_model.analysis.metrics import Metric, Analyzer, metric, \
    discretize_positions, occupancy_map, tile_analysis, hist_orientations, motion_analysis, hist_time_moving, \
    subareas, mov_bouts, compute_orientations

from navigation_model.simulation.maze import Maze


class TestMetric(unittest.TestCase):

    def test_metric(self):

        def fun1(a, b):
            return a + b

        def fun2(b, c):
            return b * c

        m1 = Metric(fun1, requirements=["a", "b"], products=["c"])
        m2 = Metric(fun2, requirements=["b", "c"], products=["d"])

        args = {"a": 1, "b": 2}

        self.assertEqual(["a", "b"], m1.requirements)
        self.assertIs(None, m1.products)
        self.assertEqual(m1(a=1, b=2), fun1(1, 2))
        self.assertTrue(m1.is_callable(args.keys()))
        self.assertEqual(m1(**args), fun1(**args))
        self.assertEqual({"c": fun1(**args)}, m1.products)

        self.assertEqual(m2(b=1, c=2), fun2(1, 2))
        self.assertFalse(m2.is_callable(args.keys()))
        self.assertRaises(TypeError, m2, **args)

    def test_decorator(self):

        @metric("c")
        def fun1(a, b):
            return a + b

        @metric("d")
        def fun2(b, c):
            return b * c

        @metric("e")
        def fun3(b, c=2):
            return b * c

        m1 = Metric(fun1, fun1.requirements, fun1.products)
        m2 = Metric(fun2, fun2.requirements, fun2.products)
        m3 = Metric(fun3, fun3.requirements, fun3.products, fun3.defaults)

        args = {"a": 1, "b": 2}

        self.assertEqual(["a", "b"], m1.requirements)
        self.assertIs(None, m1.products)
        self.assertEqual(m1(a=1, b=2), fun1(1, 2))
        self.assertTrue(m1.is_callable(args.keys()))
        self.assertEqual(m1(**args), fun1(**args))
        self.assertEqual({"c": fun1(**args)}, m1.products)

        self.assertEqual(m2(b=1, c=2), fun2(1, 2))
        self.assertFalse(m2.is_callable(args.keys()))
        self.assertRaises(TypeError, m2, **args)

        self.assertEqual(m3(b=1, c=2), fun3(1, 2))
        self.assertTrue(m3.is_callable(args.keys()))
        self.assertEqual(m3(b=2), fun3(b=2))
        self.assertEqual({"e": fun3(b=2)}, m3.products)


class TestAnalyzer(unittest.TestCase):

    def test_correct(self):

        @metric("c")
        def fun1(a, b):
            return a + b

        @metric("d")
        def fun2(b, c, e=1):
            return b * c * e

        ana = Analyzer(fun1, fun2)
        res = ana.analize(a=1, b=2)
        self.assertEqual({"a": 1, "b": 2, "c": 3, "d": 6}, res)
        self.assertEqual(res, ana.last)

    def test_error(self):

        @metric("c")
        def fun1(a, b):
            return a + b

        @metric("d")
        def fun2(b, c):
            return b * c

        @metric("f")
        def fun3(d, e):
            return d ** e

        ana = Analyzer(fun1, fun2, fun3)
        self.assertRaises(RuntimeError, ana.analize, a=1, b=2)
        self.assertEqual({'a': 1, 'b': 2, 'e': 3, 'c': 3, 'd': 6, 'f': 216}, ana.analize(a=1, b=2, e=3))


class TestMetricFunctions(unittest.TestCase):

    def setUp(self):
        self.maze = Maze.from_file("data/maze_layout.txt", size_tile=0.111)

    def test_discretize_positions(self):
        pos_list = [(0.1, 0.1), (0.35, 0.47)]
        disc_list = discretize_positions(pos_list, self.maze)
        self.assertEqual(len(disc_list), len(pos_list))
        self.assertEqual(disc_list[0], (0, 0))
        self.assertEqual(disc_list[1], (3, 4))

    def test_occupancy_map(self):
        disc_list = [(0, 0), (3, 4), (0, 0)]
        occupancy = occupancy_map(disc_list, self.maze)
        self.assertEqual(np.sum(occupancy), len(disc_list))
        self.assertEqual(occupancy[0, 0], 2)
        self.assertEqual(occupancy[3, 4], 1)

    def test_tile_analysis(self):
        disc_list = [(0, 0), (1, 1), (0, 0)]
        perc, ratios, counts = tile_analysis(disc_list, self.maze,
                                             [self.maze.MazeTile.CORNER,
                                              self.maze.MazeTile.CENTRE,
                                              self.maze.MazeTile.WALL])
        npt.assert_allclose(perc[self.maze.MazeTile.CORNER], 100 * 2 / 3)
        npt.assert_allclose(perc[self.maze.MazeTile.CENTRE], 100 * 1 / 3)
        npt.assert_allclose(perc[self.maze.MazeTile.WALL], 0)

        totals = self.maze.description()
        npt.assert_allclose(ratios[self.maze.MazeTile.CORNER], 2 / totals[self.maze.MazeTile.CORNER])
        npt.assert_allclose(ratios[self.maze.MazeTile.CENTRE], 1 / totals[self.maze.MazeTile.CENTRE])
        npt.assert_allclose(ratios[self.maze.MazeTile.WALL], 0)

        npt.assert_allclose(counts[self.maze.MazeTile.CORNER], 2)
        npt.assert_allclose(counts[self.maze.MazeTile.CENTRE], 1)
        npt.assert_allclose(counts[self.maze.MazeTile.WALL], 0)

    def test_hist_orientations(self):
        # check sizes
        self.assertRaises(RuntimeError, hist_orientations, [], [1])

        # normal execution
        ori = [0., -3.0, 0.0]
        dpos = range(len(ori))
        ho, ro, hob = hist_orientations(ori, dpos)
        npt.assert_allclose(ho, np.array([0, 0, 0, 0, 0, 0, 0, 2]))
        npt.assert_allclose(hob, np.linspace(-0.875 * np.pi, 1.125 * np.pi, 9))
        npt.assert_allclose(ro, [-3.0 + 2 * np.pi, 3.0])

        # no movement
        ori = np.random.rand(5)
        dpos = [2] * len(ori)
        ho, ro, hob = hist_orientations(ori, dpos)
        npt.assert_allclose(ho, np.zeros(8))
        npt.assert_allclose(hob, np.linspace(-0.875 * np.pi, 1.125 * np.pi, 9))
        self.assertEqual(0, len(ro))

        # bins created from possible actions
        possible_actions = [(1, 0),
                            (2, 0),
                            (-1, 0),
                            (0, -1),
                            (0, 1)]
        ori = [0., -3.0, 0.0]
        dpos = range(len(ori))
        ho, ro, hob = hist_orientations(ori, dpos, possible_actions=possible_actions)
        npt.assert_allclose(ho, np.array([0, 0, 0, 2]))
        npt.assert_allclose(ro, [-3.0 + 2 * np.pi, 3.0])
        npt.assert_allclose(hob, np.linspace(-3/4 * np.pi, 5/4 * np.pi, 5))

    def test_count_moving(self):
        pos_list = [(0, 0), (1, 1), (0, 0), (0, 0)]
        count_moving, moving_bool_list = motion_analysis(pos_list)
        self.assertEqual(count_moving, 2)
        self.assertEqual(moving_bool_list, [False, True, True, False])

    def test_hist_time_moving(self):
        moving_bool_list = [False, True, True, False]
        time_moving_histogram = hist_time_moving(moving_bool_list)
        self.assertEqual(time_moving_histogram, [(1, 0), (2, 1), (1, 0)])

    def test_subareas(self):
        pos_list = [(0, 0), (1, 5), (2, 5), (7, 7)]
        subareas_histogram = subareas(pos_list, self.maze)
        npt.assert_allclose(subareas_histogram, [0, 1, 2, 0, 0, 1, 0, 0])

    def test_mov_bouts(self):
        hist = [(1, 0), (2, 1), (1, 0)]
        mov, stop = mov_bouts(hist)
        self.assertEqual(mov, 2)
        self.assertEqual(stop, 0)

    def test_compute_orientations(self):
        pos_list = [(0, 0), (0, 0), (0, 1), (1, 1)]
        ori = compute_orientations(pos_list)
        self.assertEqual(len(pos_list), len(ori))
        npt.assert_allclose(ori, [np.pi / 2] * 3 + [0.0])
