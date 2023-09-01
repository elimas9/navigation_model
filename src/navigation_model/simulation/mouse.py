import numpy as np
import copy
from matplotlib.patches import Polygon

from navigation_model.visualization.plotting import plot_trajectory

from _navigation_model import Mouse as MouseBase

class Mouse(MouseBase):

    def __init__(self, init_pos, init_ori, possible_actions, save_history=True):
        MouseBase.__init__(self, init_pos, init_ori, possible_actions, save_history)

    @property
    def history(self):
        return self.history_position, self.history_orientation

    def plot(self, ax, endpoints=False, c="y", size=0.1, endpoint_size=None):
        """
        Plot the current position of the mouse and its endpoints

        :param ax: axis where to plot
        :param endpoints: if True plot the current endpoints
        :param c: color
        :param size: size of the mouse (same units as the maze tile size)
        :param endpoint_size: size of the endpoints
        """
        mouse_points = np.array([[size * 2/3, 0.], [-size * 1/3, size/3], [-size * 1/3, -size/3]])
        rot = np.array([[np.cos(self.orientation), -np.sin(self.orientation)], [np.sin(self.orientation), np.cos(self.orientation)]])
        mouse_points = np.array([rot @ mp for mp in mouse_points]) + self.position
        p = Polygon(mouse_points, color=c, zorder=3)
        ax.add_patch(p)
        # endpoints
        if endpoints:
            ep = self.get_endpoints()
            ep_x = [p[0] for p in ep]
            ep_y = [p[1] for p in ep]
            ax.scatter(ep_x, ep_y, marker='x', c=c, zorder=3, s=endpoint_size)

    def plot_history(self, ax, c="y"):
        """
        Plot the history as a trajectory

        :param ax: axis where to plot
        :param c: color
        """
        plot_trajectory(ax, self.history_position, c=c)
