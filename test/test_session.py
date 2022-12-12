from navigation_model.analysis.data import Session, SessionList

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

    def test_nan(self):
        times = np.linspace(0., 1., 10)
        traj = np.random.random((10, 2))
        traj[0] = (np.nan, np.nan)

        session = Session(times, traj, new_sampling_time=0.2)
        expected = traj[::2]
        expected[0] = traj[1]
        npt.assert_allclose(session.trajectory, expected)


class TestSessionList(unittest.TestCase):

    def test_session_list(self):
        slist = SessionList()
        slist.create(np.zeros(10), list(zip(np.arange(0, 10), np.arange(0, 10))), reward=list(range(0, 10)))
        slist.create(np.zeros(10), list(zip(np.arange(10, 20), np.arange(10, 20))), reward=list(range(10, 20)))
        npt.assert_allclose(slist.all_trajectories, list(zip(np.arange(0, 20), np.arange(0, 20))))
        npt.assert_allclose(slist.all_trajectories, slist[0].trajectory + slist[1].trajectory)
        npt.assert_allclose(slist.all_orientations, slist[0].orientations + slist[1].orientations)
        npt.assert_allclose(slist.all_rewards, slist[0].reward + slist[1].reward)

    def test_save_load(self):
        sl1 = SessionList()
        sl1.create(np.zeros(10), list(zip(np.arange(0, 10), np.arange(0, 10))), reward=list(range(0, 10)))
        sl1.create(np.zeros(10), list(zip(np.arange(10, 20), np.arange(10, 20))), reward=list(range(10, 20)))

        sl_list = sl1.to_list()
        self.assertEqual(sl_list[0], sl1[0].to_dict())
        self.assertEqual(sl_list[1], sl1[1].to_dict())

        sl2 = SessionList.from_list(sl_list)

        npt.assert_allclose(sl2.all_trajectories, sl1.all_trajectories)
        npt.assert_allclose(sl2.all_trajectories, sl1.all_trajectories)
        npt.assert_allclose(sl2.all_orientations, sl1.all_orientations)
        npt.assert_allclose(sl2.all_rewards, sl1.all_rewards)

    def test_creation(self):
        s = Session(np.zeros(10), list(zip(np.arange(0, 10), np.arange(0, 10))))

        sl = SessionList([])
        self.assertEqual(len(sl), 0)

        sl = SessionList(s)
        self.assertEqual(len(sl), 1)

        sl = SessionList([s, s])
        self.assertEqual(len(sl), 2)

        sl = SessionList(s, s)
        self.assertEqual(len(sl), 2)

        self.assertRaises(RuntimeError, SessionList, [s, 1])
        self.assertRaises(RuntimeError, SessionList, s, 2)

        self.assertRaises(RuntimeError, sl.append, 2)

    def test_indexing(self):
        slist = SessionList()
        for _ in range(10):
            slist.create(np.zeros(10), list(zip(np.arange(0, 10), np.arange(0, 10))), reward=list(range(0, 10)))

        self.assertEqual(type(slist[0]), Session)
        sl = slist[5:7]
        self.assertEqual(type(sl), SessionList)
        self.assertEqual(len(sl), 2)

    def test_add_mul(self):
        sl1 = SessionList()
        for _ in range(10):
            sl1.create(np.zeros(10), list(zip(np.arange(0, 10), np.arange(0, 10))), reward=list(range(0, 10)))
        sl2 = SessionList()
        for _ in range(5):
            sl2.create(np.zeros(10), list(zip(np.arange(0, 10), np.arange(0, 10))), reward=list(range(0, 10)))

        sl3 = sl1 + sl2
        self.assertEqual(len(sl3), 15)

        sl4 = sl1 * 3
        self.assertEqual(len(sl4), 30)

        sl5 = 4 * sl2
        self.assertEqual(len(sl5), 20)

    def test_flatten(self):
        slist = SessionList()
        slist.create(np.zeros(10), list(zip(np.arange(0, 10), np.arange(0, 10))), reward=list(range(0, 10)))
        slist.create(np.zeros(10), list(zip(np.arange(10, 20), np.arange(10, 20))), reward=list(range(10, 20)))
        s = slist.flatten()
        npt.assert_allclose(s.trajectory, slist[0].trajectory + slist[1].trajectory)
        npt.assert_allclose(s.reward, slist[0].reward + slist[1].reward)
        npt.assert_allclose(s.sampling_time, slist[0].sampling_time)
        npt.assert_allclose(s.orientations, slist[0].orientations + slist[1].orientations)
