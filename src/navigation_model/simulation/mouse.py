import numpy as np
from matplotlib.patches import Polygon

from navigation_model.visualization.plotting import plot_trajectory


class Mouse:

    def __init__(self, init_pos, init_ori, possible_actions, save_history=True):
        self._init_pos = init_pos
        self._pos = init_pos
        self._init_ori = init_ori
        self._ori = init_ori
        self._possible_actions = np.array(possible_actions)
        self._save_history = save_history
        self._history_pos = None
        self._history_ori = None
        self.reset_history()

    @property
    def position(self):
        return self._pos

    @property
    def initial_position(self):
        return self._init_pos

    @property
    def orientation(self):
        return self._ori

    @property
    def initial_orientation(self):
        return self._init_ori

    @property
    def history(self):
        return self.history_position, self.history_orientation

    @property
    def history_position(self):
        return self._history_pos

    @property
    def history_orientation(self):
        return self._history_ori

    def enable_history(self, enable=True):
        """
        Toggle history saving

        Also resets history.

        :param enable: True to enable history, false otherwise
        """
        self._save_history = enable
        self.reset_history()

    def reset(self, initial_pos=None, initial_ori=None):
        """
        Resets position and orientation to the initial ones

        Also resets history.

        Can also change initial position and orientation.

        :param initial_pos: change initial position
        :param initial_ori: change initial orientation
        """
        if initial_pos is not None:
            self._init_pos = initial_pos
        if initial_ori is not None:
            self._init_ori = initial_ori
        self._pos = self._init_pos
        self._ori = self._init_ori
        self.reset_history()

    def reset_history(self):
        """
        Resets history
        """
        if self._save_history:
            self._history_pos = [self._init_pos]
            self._history_ori = [self._init_ori]
        else:
            self._history_pos = []
            self._history_ori = []

    def get_endpoints(self):
        """
        Compute a list of possible endpoints for the mouse, with absolute coordinates

        :return: list of np.array of endpoints (2 coordinates)
        """
        # compute rotation matrix (rotation matrix for counterclockwise rotation)
        rot = np.array([[np.cos(self._ori), -np.sin(self._ori)], [np.sin(self._ori), np.cos(self._ori)]])

        ends = np.zeros((len(self._possible_actions), 2))
        for idx, act in enumerate(self._possible_actions):
            ends[idx] = self._pos + np.matmul(rot, act)
        return ends

    def move_to(self, x, y=None):
        """
        Set the new position and compute the orientation from the direction of movement

        :param x: x of the new absolute continuous position (or pair of coordinates)
        :param y: y of the new absolute continuous position (or None)
        :return: True if the mouse has moved, False otherwise
        """
        if y is None:
            new_pos = x
        else:
            new_pos = (x, y)
        moving = False
        if not np.all(np.isclose(new_pos, self._pos)):
            self._ori = np.arctan2(new_pos[1] - self._pos[1], new_pos[0] - self._pos[0])
            self._pos = new_pos
            moving = True
        if self._save_history:
            self._history_pos.append(self._pos)
            self._history_ori.append(self._ori)
        return moving

    def plot(self, ax, endpoints=False, c="y", size=0.1):
        """
        Plot the current position of the mouse and its endpoints

        :param ax: axis where to plot
        :param endpoints: if True plot the current endpoints
        :param c: color
        :param size: size of the mouse (same units as the maze tile size)
        """
        mouse_points = np.array([[size * 2/3, 0.], [-size * 1/3, size/3], [-size * 1/3, -size/3]])
        rot = np.array([[np.cos(self._ori), -np.sin(self._ori)], [np.sin(self._ori), np.cos(self._ori)]])
        mouse_points = np.array([rot @ mp for mp in mouse_points]) + self.position
        p = Polygon(mouse_points, color=c, zorder=3)
        ax.add_patch(p)
        # endpoints
        if endpoints:
            ep = self.get_endpoints()
            ep_x = [p[0] for p in ep]
            ep_y = [p[1] for p in ep]
            ax.scatter(ep_x, ep_y, marker='x', c=c, zorder=3)

    def plot_history(self, ax, c="y"):
        """
        Plot the history as a trajectory

        :param ax: axis where to plot
        :param c: color
        """
        plot_trajectory(ax, self.history_position, c=c)
