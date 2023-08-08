import numpy as np

from navigation_model.analysis.metrics import compute_orientations
from navigation_model.visualization.plotting import plot_trajectory, plot_reward_sequence


class Session:
    """
    An experimental session
    """

    def __init__(self, timestamps, trajectory, new_sampling_time=None, reward=None, first=None, end=None, interval=None,
                 maze=None, possible_actions=None):
        """
        Create the session and do an optional subsampling of the data

        :param timestamps: list of timestamps
        :param trajectory: list of coordinates
        :param new_sampling_time: sampling time for subsampling
        :param reward: list of rewards
        :param first: integer indicating the first simulation steps to take into account in the creation of the
        trajectory
        :param end: integer indicating the final simulation steps to take into account in the creation of the trajectory
        :param interval: tuple or list indicating the starting and final simulation steps to take into account in the
        creation of the trajectory
        :param maze: maze object
        :param possible_actions: list of tuples describing the next relative possible actions in the model
        """
        if len(trajectory) != len(timestamps):
            raise RuntimeError("Session: time and trajectory must have the same lenght")
        if reward is not None and len(reward) != len(timestamps):
            raise RuntimeError("Session: reward must have the same length as time and trajectory")
        self._trajectory = []
        self._reward = None if reward is None else []
        if new_sampling_time is None:
            self._trajectory = trajectory
            self._reward = reward
            self._sampling_t = np.mean(np.diff(timestamps))
        else:
            self._sampling_t = new_sampling_time
            idx = 0
            n_samples = len(timestamps)
            while idx < n_samples - 1:
                # advance until at least sampling_t seconds have passed
                curr_time = timestamps[idx]
                if np.isnan(curr_time):
                    idx += 1
                    continue
                next_idx = idx + 1
                while next_idx < n_samples and \
                        (np.isnan(timestamps[next_idx]) or timestamps[next_idx] <= curr_time + self._sampling_t):
                    next_idx += 1

                # set trajectory if valid
                traj_window = np.array(trajectory[idx:next_idx])
                traj_window = traj_window[np.all(np.isfinite(traj_window), axis=1)]
                if len(traj_window) > 0:
                    self._trajectory.append(traj_window[0])
                else:
                    idx = next_idx
                    continue

                # set reward
                if reward is not None:
                    reward_window = np.array(reward[idx:next_idx])
                    reward_window = reward_window[reward_window != 0]
                    reward_window = reward_window[~np.isnan(reward_window)]
                    if len(reward_window) > 0:  # here we assume that all rewards are equal
                        r = reward_window[0]
                    else:
                        r = 0.0
                    self._reward.append(r)

                idx = next_idx

        if first is not None:
            self._trajectory = self._trajectory[:first]

        if end is not None:
            self._trajectory = self._trajectory[-end:]

        if interval is not None:
            self._trajectory = self._trajectory[interval[0]:interval[1]]

        self._orientations = None
        self._compute_orientations(maze=maze, possible_actions=possible_actions)

    @property
    def trajectory(self):
        """
        Get the sampled trajectory

        :return: trajectory
        """
        return self._trajectory

    @property
    def has_reward(self):
        """
        Check if the session has a reward

        :return: True if there is a reward
        """
        return self._reward is not None

    @property
    def reward(self):
        """
        Get the sampled reward

        :return: reward
        """
        return self._reward

    @property
    def orientations(self):
        """
        Get the absolute orientations

        :return: orientations
        """
        return self._orientations

    @property
    def sampling_time(self):
        """
        Get the sampling time

        :return: sampling time
        """
        return self._sampling_t

    @property
    def initial_position(self):
        """
        Initial position of the session

        :return: initial position
        """
        return self._trajectory[0]

    @property
    def initial_orientation(self):
        """
        Initial orientation of thes session

        :return: initial orientation
        """
        return self._orientations[0]

    def to_dict(self):
        """
        Transform the session in a dictionary

        :return: dictionary
        """
        d = {
            "sampling_time": self._sampling_t,
            "trajectory": self._trajectory,
        }
        if self._reward is not None:
            d["reward"] = self._reward
        return d

    @classmethod
    def from_dict(cls, session_dict):
        """
        Create a new session from a dictionary

        :param session_dict: dictionary with trajectory, sampling_time and (optional) reward keys
        :return: new Session object
        """
        s = cls.__new__(cls)
        s._trajectory = session_dict["trajectory"]
        s._sampling_t = session_dict["sampling_time"]
        s._reward = session_dict.get("reward", None)
        s._orientations = None
        s._compute_orientations()
        return s

    def plot(self, ax, c_trajectory="b", c_reward="r", marker_reward="x"):
        """
        Plot trajectory and reward events

        :param ax: axis where to plot
        :param c_trajectory: color of the trajectory
        :param c_reward: color of the reward events
        :param marker_reward: marker for the reward events
        """
        plot_trajectory(ax, self._trajectory, c=c_trajectory)
        plot_reward_sequence(ax, np.array(self._reward), np.array(self._trajectory), c=c_reward, marker=marker_reward)

    def _compute_orientations(self, maze=None, possible_actions=None):
        """
        Compute the orientations

        :param maze: maze object
        :param possible_actions: list of tuples describing the next relative possible actions in the model
        """
        self._orientations = compute_orientations(self._trajectory, maze=maze, possible_actions=possible_actions)

    def __len__(self):
        return len(self._trajectory)


class SessionList:
    """
    An iterable list of sessions that can give aggregate session information
    """

    def __init__(self, *args):
        """
        Create a new session list

        :param args: sequence of sessions (or list of sessions)
        """
        if len(args) == 1 and type(args[0]) is list:
            self._sessions = args[0]
        else:
            self._sessions = [*args]
        # check that we are only storing sessions
        for s in self._sessions:
            self._assert_session(s)

    def create(self, timestamps, trajectory, new_sampling_time=None, reward=None, first=None, end=None, interval=None,
               maze=None, possible_actions=None):
        """
        Create a new session and append it to the list

        :param timestamps: list of timestamps
        :param trajectory: list of coordinates
        :param new_sampling_time: sampling time for subsampling
        :param reward: list of rewards
        :param first: integer indicating the first simulation steps to take into account in the creation of the
        trajectory
        :param end: integer indicating the final simulation steps to take into account in the creation of the trajectory
        :param interval: tuple or list indicating the starting and final simulation steps to take into account in the
        creation of the trajectory
        :param maze: maze object
        :param possible_actions: list of tuples describing the next relative possible actions in the model
        """
        self._sessions.append(Session(timestamps, trajectory, new_sampling_time, reward, first, end, interval, maze,
                                      possible_actions))

    def append(self, s):
        """
        Append a session to the list

        :param s: session
        """
        self._assert_session(s)
        self._sessions.append(s)

    @property
    def all_trajectories(self):
        """
        Get all trajectory as a single list

        :return: trajectories
        """
        all_data = []
        for s in self._sessions:
            all_data += s.trajectory
        return all_data

    @property
    def all_rewards(self):
        """
        Get all rewards as a single list

        :return: rewards
        """
        all_data = []
        for s in self._sessions:
            if s.has_reward:
                all_data += s.reward
            else:
                return None
        return all_data

    @property
    def all_orientations(self):
        """
        Get all orientations as a single list

        :return: orientations
        """
        all_data = []
        for s in self._sessions:
            all_data += s.orientations
        return all_data

    def __getitem__(self, i):
        sess = self._sessions[i]
        if isinstance(i, slice):
            return SessionList(sess)
        return sess

    def __iter__(self):
        return iter(self._sessions)

    def __len__(self):
        return len(self._sessions)

    def to_list(self):
        """
        Transform the session list in a list

        :return: list
        """
        return [s.to_dict() for s in self._sessions]

    @classmethod
    def from_list(cls, sessions_list):
        """
        Create a new session list from a list

        :param sessions_list: list of dictionaries with trajectory, sampling_time and (optional) reward keys
        :return: new SessionList object
        """
        sl = cls()
        for sd in sessions_list:
            sl._sessions.append(Session.from_dict(sd))
        return sl

    @staticmethod
    def _assert_session(s):
        if type(s) is not Session:
            raise RuntimeError("SessionList can only store objects of type Session")

    def __add__(self, other):
        """
        :type other: SessionList
        :rtype SessionList:
        """
        if type(other) is not SessionList:
            raise RuntimeError("Can only add another SessionList to this SessionList")
        return SessionList(self._sessions + other._sessions)

    def __mul__(self, times):
        """
        :type times: int
        :rtype SessionList:
        """
        if type(times) is not int:
            raise RuntimeError("Can only multiply SessionList to an int")
        return SessionList(self._sessions * times)

    def __rmul__(self, times):
        """
        :type times: int
        :rtype SessionList:
        """
        if type(times) is not int:
            raise RuntimeError("Can only multiply SessionList to an int")
        return SessionList(self._sessions * times)

    def flatten(self):
        """
        Flatten the SessionList into a single Session

        :return: Session
        """
        news = Session.__new__(Session)
        news._trajectory = self.all_trajectories
        news._reward = self.all_rewards
        news._sampling_t = np.mean([s.sampling_time for s in self._sessions])
        news._orientations = self.all_orientations
        return news


def adjust_positions_to_maze(continuous_positions, maze):
    """
    Adjust positions to fit in a maze

    Positions outsite the maze are changed with the center of the closest tile

    :param continuous_positions: list of continuous positions
    :param maze: maze
    :return: adjusted list of positions
    """
    adjusted_pos = []
    for p in continuous_positions:
        if not np.any(np.isnan(p)) and not maze.is_visitable_cont(p):
            adjusted_pos.append(maze.disc2cont(maze.get_closest_visitable_cont(p)))
        else:
            adjusted_pos.append(p)
    return adjusted_pos
