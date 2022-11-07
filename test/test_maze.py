from navigation_model.simulation.maze import Maze

import numpy.testing as npt
import unittest


class TestMaze(unittest.TestCase):

    def setUp(self):

        mazestr = """CWWWWWC
WMMMMMW
WMWWWMW
WMWVWMW
WMWWWMW
WMMMMMW
CWWWWWC"""
        subareas = """1111111
1111111
2222222
2220222
3333333
3333333"""
        self.maze = Maze(mazestr, subareas, size_tile=0.111)

    def test_properties(self):
        descr_tile = self.maze.size_tile
        self.assertEqual(descr_tile, 0.111)
        shape = self.maze.shape
        self.assertEqual(shape, (7, 7))

    def test_descr_maze(self):
        descr = self.maze.description()
        self.assertEqual(descr[self.maze.MazeTile.CORNER], 4)
        self.assertEqual(descr[self.maze.MazeTile.WALL], 28)
        self.assertEqual(descr[self.maze.MazeTile.CENTRE], 16)

    def test_cont2disc(self):
        pos = (0.443, 0.221)
        pos_disc = self.maze.cont2disc(pos)
        pos_abs_centre = self.maze.get_tile_centre(pos_disc)
        self.assertEqual(pos_disc, (3, 1))
        npt.assert_allclose(pos_abs_centre, (0.3885, 0.1665))

        poss = [(0.443, 0.221)]
        poss_disc = self.maze.cont2disc_list(poss)
        pos_abs_centre = self.maze.get_tile_centre(poss_disc[0])
        self.assertEqual(pos_disc, (3, 1))
        npt.assert_allclose(pos_abs_centre, (0.3885, 0.1665))

    def test_disc2cont(self):
        pos = (3, 5)
        pos_cont = self.maze.disc2cont(pos)
        npt.assert_allclose(pos_cont, (0.3885, 0.6105))

        poss = [(3, 5)]
        poss_cont = self.maze.disc2cont_list(poss)
        npt.assert_allclose(poss_cont, [(0.3885, 0.6105)])

    def test_getters(self):
        self.assertEqual(self.maze.MazeTile.CORNER, self.maze.get_tile(0, 0))
        self.assertEqual([self.maze.MazeTile.CORNER, self.maze.MazeTile.VOID], self.maze.get_tiles([(6, 6), (3, 3)]))
        self.assertEqual(1, self.maze.get_subarea(0, 0))
        self.assertEqual([1, 0], self.maze.get_subareas([(0, 0), (3, 3)]))

    def test_get_corners(self):
        corners = self.maze.get_coordinates_of_type(self.maze.MazeTile.CORNER)
        self.assertEqual([(0, 0), (0, 6), (6, 0), (6, 6)], corners)

    def test_get_number_visitable_tiles(self):
        n_visit_table = self.maze.get_number_visitable_tiles()
        self.assertEqual(n_visit_table, 48)

    def test_closest_tile(self):
        p = self.maze.get_closest_visitable(-1, -1)
        self.assertEqual(p, (0, 0))

    def test_movement_possible(self):
        self.assertTrue(self.maze.is_movement_possible(0, 0, 0, 6))
        self.assertFalse(self.maze.is_movement_possible(3, 0, 3, 6))

    def test_visitable(self):
        self.assertTrue(self.maze.is_visitable(1, 1))
        self.assertFalse(self.maze.is_visitable(3, 3))
        self.assertEqual(self.maze.are_visitable([(1, 1), (2, 2)]), [True, True])
        self.assertEqual(self.maze.are_visitable([(1, 1), (3, 3)]), [True, False])


class TestMazeFile(unittest.TestCase):

    def test_load(self):
        maze = Maze.from_file("data/maze_layout.txt", size_tile=0.111)
        self.assertEqual(9, maze.size_x)
        self.assertEqual(9, maze.size_y)

    def test_no_subareas(self):
        maze = Maze.from_file("data/maze_layout_no_subareas.txt", size_tile=0.111)
        self.assertRaises(RuntimeError, maze.get_subareas, (0, 0))


if __name__ == '__main__':
    unittest.main()
