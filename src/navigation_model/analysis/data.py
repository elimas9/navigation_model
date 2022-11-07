import numpy as np

from navigation_model.analysis.metrics import compute_orientations
from navigation_model.visualization.plotting import plot_trajectory, plot_reward_sequence


class Session:
    """
    An experimental session
    """

    def __init__(self, timestamps, trajectory, new_sampling_time=None, reward=None):
        """
        Create the session and do an optional subsampling of the data

        :param timestamps: list of timestamps
        :param trajectory: list of coordinates
        :param new_sampling_time: sampling time for subsampling
        :param reward: list of rewards
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
                # set trajectory
                self._trajectory.append(trajectory[idx])

                # advance until at least sampling_t seconds have passed
                curr_time = timestamps[idx]
                next_idx = idx + 1
                while next_idx < n_samples and timestamps[next_idx] <= curr_time + self._sampling_t:
                    next_idx += 1

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
        self._orientations = None
        self._compute_orientations()

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

    def _compute_orientations(self):
        self._orientations = compute_orientations(self._trajectory)
