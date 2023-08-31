import numpy as np
import unittest
import numpy.testing as npt

from navigation_model.simulation.mouse import Mouse


class TestMouse(unittest.TestCase):

    def setUp(self) -> None:
        self.init_pos = (1, 2)  # tuple with initial position of the mouse
        self.init_ori = np.pi / 2  # initial rotation (in radians) -- 0.35 rad ~ 20 degrees
        possible_actions = [(1, 0), (-1, 0)]  # list of tuples representing relative displacements

        self.m = Mouse(self.init_pos, self.init_ori, possible_actions)

    def test_description(self):
        npt.assert_allclose(self.m.position, self.init_pos)
        self.assertEqual(self.m.orientation, self.init_ori)

    def test_endpoints(self):
        m_end_points = self.m.get_endpoints()
        self.assertEqual(2, len(m_end_points))
        self.assertTrue(np.all(np.isclose(m_end_points[0], np.array((1, 3)))))
        self.assertTrue(np.all(np.isclose(m_end_points[1], np.array((1, 1)))))

    def test_move(self):
        new_pos = (2, 2)
        moved = self.m.move_to(new_pos)
        self.assertTrue(moved)
        npt.assert_allclose(self.m.position, new_pos)
        self.assertEqual(self.m.orientation, 0)

        new_pos = (0, 2)
        moved = self.m.move_to(*new_pos)
        self.assertTrue(moved)
        npt.assert_allclose(self.m.position, new_pos)
        self.assertAlmostEqual(self.m.orientation, np.pi)

    def test_reset(self):
        self.m.move_to(1, 1)
        self.assertFalse(np.allclose(self.m.position, self.init_pos))
        self.assertNotEqual(self.m.orientation, self.init_ori)
        self.m.reset()
        npt.assert_allclose(self.m.position, self.init_pos)
        self.assertEqual(self.m.orientation, self.init_ori)

    def test_history(self):
        self.m.move_to(1, 1)
        npt.assert_allclose(self.m.history_position, [(1, 2), (1, 1)])
        self.m.move_to(2, 2)
        npt.assert_allclose(self.m.history_position, [(1, 2), (1, 1), (2, 2)])
        self.m.reset_history()
        npt.assert_allclose(self.m.history_position, [(1, 2)])

        self.m.enable_history(False)
        self.m.move_to(2, 2)
        self.assertEqual(self.m.history_position, [])
        self.assertEqual(self.m.history_orientation, [])
