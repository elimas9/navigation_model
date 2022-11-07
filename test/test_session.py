from navigation_model.analysis.data import Session

import unittest
import numpy as np
import numpy.testing as npt


class TestSession(unittest.TestCase):

    def test_session_nosampling(self):
        times = np.linspace(0., 1., 10)
        traj = np.random.random((10, 2))
        reward = np.random.random(10)

        # with reward
        session = Session(times, traj, reward=reward)
        npt.assert_allclose(session.sampling_time, 1 / 9)
        npt.assert_allclose(session.trajectory, traj)
        npt.assert_allclose(session.reward, reward)
        npt.assert_allclose(session.initial_position, traj[0])
        npt.assert_allclose(session.initial_orientation,
                            np.arctan2(traj[1][1] - traj[0][1], traj[1][0] - traj[0][0]))

        # no reward
        session = Session(times, traj)
        npt.assert_allclose(session.sampling_time, 1 / 9)
        npt.assert_allclose(session.trajectory, traj)
        self.assertIsNone(session.reward)

    def test_session_sampling(self):
        times = np.linspace(0., 1., 10)
        traj = np.random.random((10, 2))
        reward = np.random.random(10)

        # with reward
        session = Session(times, traj, reward=reward, new_sampling_time=0.2)
        npt.assert_allclose(session.trajectory, traj[::2])
        npt.assert_allclose(session.reward, reward[::2])

        # no reward
        session = Session(times, traj, new_sampling_time=0.2)
        npt.assert_allclose(session.trajectory, traj[::2])
        self.assertIsNone(session.reward)

    def test_save_load(self):
        times = np.linspace(0., 1., 10)
        traj = np.random.random((10, 2))
        reward = np.random.random(10)

        # create dict
        s1 = Session(times, traj, reward=reward)
        d = s1.to_dict()
        npt.assert_allclose(d["sampling_time"], 1 / 9)
        npt.assert_allclose(d["trajectory"], traj)
        npt.assert_allclose(d["reward"], reward)

        # laod from dictionary
        s2 = Session.from_dict(d)
        npt.assert_allclose(s1.trajectory, s2.trajectory)
        npt.assert_allclose(s1.sampling_time, s2.sampling_time)
        npt.assert_allclose(s1.orientations, s2.orientations)
        npt.assert_allclose(s1.reward, s2.reward)
