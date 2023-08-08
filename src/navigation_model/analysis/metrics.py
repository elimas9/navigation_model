import inspect

import numpy as np
import copy
import math


class Metric:
    """
    Metric class

    A metric is a callable object with input requirements and that pruduces output products (measures).
    """

    def __init__(self, fun, requirements=None, products=None, defaults=None):
        """
        Create a new metric

        :param fun: callable to compute the metric
        :param requirements: list of requirements
        :param products: list of outputs
        """
        self._fun = fun
        self._requirements = requirements
        if self._requirements is None:
            self._requirements = []
        self._product_labels = products
        if self._product_labels is None:
            self._product_labels = []
        self._defaults = defaults
        if self._defaults is None:
            self._defaults = dict()

        self._products = None

    def __call__(self, **kwargs):
        """
        Calls the callable

        Calls the underlying callable and return its output. Also store the products.
        """
        fun_inputs = self._get_inputs(kwargs)
        res = self._fun(**fun_inputs)
        if type(res) != tuple:
            _res = (res, )
        else:
            _res = res
        self._products = {lab: pr for lab, pr in zip(self._product_labels, _res)}
        return res

    def _get_inputs(self, kwargs):

        fun_kwargs = {}
        for r in self._requirements:
            if r in self._defaults:
                fun_kwargs[r] = self._defaults[r]
            if r in kwargs:
                fun_kwargs[r] = kwargs[r]
        return fun_kwargs

    def is_callable(self, available):
        """
        Chech if the metric is callable, given a list of available results

        :param available: list of available requirements
        :return: True if the function is callable, false otherwise
        """
        for r in self._requirements:
            if r not in available and r not in self._defaults:
                return False
        return True

    @property
    def requirements(self):
        """
        Get the requirements

        :return: list of requirements
        """
        return self._requirements

    @property
    def products(self):
        """
        Get the last products

        :return: last products generated
        """
        return self._products


def metric(*products):
    """
    Decorator to add requirements and products to a function

    This will add three properties (requirements, products and defaults) to the function,
    so that it can be passed to an Analyzer.
    Requirements and defaults are taken from the function signature.

    :param products: list of outputs
    """
    def deco(fun):
        signature = inspect.signature(fun)
        fun.requirements = list(signature.parameters.keys())
        fun.products = products
        fun.defaults = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        return fun

    return deco


class Analyzer:
    """
    Class that aggregate metrics and perform analysis

    This class will compute the metrics iteratively as soon as new products are generated by other metrics.
    """

    def __init__(self, *metrics):
        """
        Create an Analyzer

        :param metrics: list of Metrics (or decorated functions)
        """
        self._metrics = []
        for m in metrics:
            if type(m) is Metric:
                self._metrics.append(m)
            elif hasattr(m, "defaults"):
                self._metrics.append(Metric(m, m.requirements, m.products, m.defaults))
            else:
                self._metrics.append(Metric(m, m.requirements, m.products))
        self._last = None

    def analize(self, **current_measures):
        """
        Perform the analysis with all the metrics

        If some metrics cannot be called a RuntimeError will be thrown.

        :param current_measures: starting measures
        :return: dictionary with results of the analysis
        """
        to_call = list(range(len(self._metrics)))

        while len(to_call) > 0:

            for m in to_call:
                if self._metrics[m].is_callable(current_measures.keys()):
                    self._metrics[m](**current_measures)
                    current_measures.update(self._metrics[m].products)
                    to_call.remove(m)
                    break
            else:
                av = set(current_measures.keys())
                all_measures = []
                for m in self._metrics:
                    all_measures += m.requirements
                all_measures = set(all_measures)
                raise RuntimeError(f"Analysis failed: one among these measures cannot be produced: {all_measures - av}")

        self._last = current_measures

        return current_measures

    def __call__(self, **current_measures):
        """
        Calls analize

        :param current_measures: current_measures: starting measures
        :return: dictionary with results of the analysis
        """
        return self.analize(**current_measures)

    @property
    def last(self):
        """
        Get the last results of the analysis

        :return: last result
        """
        return self._last


######################
# begin actual metrics

@metric("discrete_positions")
def discretize_positions(continuous_positions, maze):
    """
    Metric that computes discrete positions from continuous ones

    :param continuous_positions: list of continuous
    :param maze: maze for the discretization
    :return: list of discrete positions
    """
    return maze.cont2disc_list(continuous_positions)


@metric("occupancy")
def occupancy_map(discrete_positions, maze):
    """
    Compute the occupancy map (number of visits for every tile)

    :param discrete_positions: list of discrete positions
    :param maze: maze
    :return: occupancy map
    """
    occupancy = np.zeros(maze.shape)
    for p in discrete_positions:
        occupancy[p[0], p[1]] += 1
    return occupancy


@metric("tiles_perc", "tiles_ratio", "tiles_count")
def tile_analysis(discrete_positions, maze, tiles=None):
    """
    Compute percentage, ratio and count for visited tile types

    :param discrete_positions: list of discrete positions
    :param maze: maze
    :param tiles: list of tile types to analyze
    :return: percentage by type, ratio by type, count by type
    """
    # get default tiles
    if tiles is None:
        tiles = list(maze.MazeTile)

    n_positions = len(discrete_positions)

    # count
    count = {t: 0 for t in tiles}
    for tile in discrete_positions:
        type_tile = maze.get_tile(tile)
        count[type_tile] += 1

    # perc
    perc = {t: count[t] * 100 / n_positions for t in tiles}

    # ratio
    totals = maze.description()
    ratio = {t: count[t] / totals[t] if totals[t] > 0 else 0 for t in tiles}

    return perc, ratio, count


def _hist_orientation_commons(caller, n_bins=None, max_lim=None, possible_actions=None):
    if n_bins is not None and possible_actions is not None:
        raise RuntimeError(f"Cannot use both n_bins and possible_actions in {caller}")
    if possible_actions is not None:
        # compute list of bin edges from possible actions
        possible_actions = np.array(possible_actions)
        angles = np.arctan2(possible_actions[:, 1], possible_actions[:, 0])
        angles = np.sort(angles)
        angles = np.unique(angles)
        bins = [(angles[-1] - 2 * np.pi + angles[0]) / 2]
        for i in range(len(angles) - 1):
            bins.append((angles[i] + angles[i + 1]) / 2)

        bins.append((angles[0] + 2 * np.pi + angles[-1]) / 2)
        max_lim = bins[-1]
        min_lim = bins[0]
    else:
        if max_lim is None:
            max_lim = np.pi * 1.125
        if n_bins is None:
            bins = 8
        else:
            bins = n_bins
        min_lim = max_lim - 2 * np.pi  # ~ -2.75 by default

    return bins, min_lim, max_lim


@metric("orientations_histogram", "relative_orientations", "orientations_histogram_bins")
def hist_orientations(absolute_orientations, discrete_positions, n_bins=None, max_lim=None, possible_actions=None,
                      distance=False):
    """
    Compute the histogram of orientations and the relative orientation (when there is motion)

    This can either produce evenly spaced bins (by specifying n_bins and max_lim) or uneven bins if possible_actions
    are given.

    :param absolute_orientations: list of absolute orientation
    :param discrete_positions: list of discrete positions
    :param n_bins: number of bins for the histogram
    :param max_lim: orientation corresponding to the last bin
    :param possible_actions: list of possible actions to create the bins
    :return: histogram of orientations, relative orientations, bins for the histogram
    """
    if len(absolute_orientations) != len(discrete_positions):
        raise RuntimeError(f"Wrong input sizes for hist_orientations {len(absolute_orientations)} != {len(discrete_positions)}")

    bins, min_lim, max_lim = _hist_orientation_commons("hist_orientations", n_bins, max_lim, possible_actions)

    relative_orientations = []
    bin_2steps = 0
    for tim in range(0, len(absolute_orientations) - 1):
        rel_ori = absolute_orientations[tim + 1] - absolute_orientations[tim]
        if rel_ori > max_lim:
            rel_ori = rel_ori - np.pi * 2
        elif rel_ori < min_lim:
            rel_ori = rel_ori - (-np.pi * 2)

        if discrete_positions[tim + 1] != discrete_positions[tim]:
            if distance:
                # if (np.histogram(rel_ori, bins=bins, range=(min_lim, max_lim))[0] == [0, 0, 0, 1, 0, 0, 0, 0]).all():
                if (np.histogram(rel_ori, bins=bins, range=(min_lim, max_lim))[0] == [0, 0, 1, 0, 0, 0]).all():
                    dist = math.dist(discrete_positions[tim], discrete_positions[tim + 1])
                    if dist > 1.5:
                        bin_2steps += 1
                    else:
                        relative_orientations.append(rel_ori)
                else:
                    relative_orientations.append(rel_ori)
            else:
                relative_orientations.append(rel_ori)

    histogram_orientations = np.histogram(relative_orientations, bins=bins, range=(min_lim, max_lim))

    if distance:
        # print(histogram_orientations[0])
        # print(np.array([bin_2steps]))

        return np.concatenate((histogram_orientations[0], np.array([bin_2steps])), axis=0), relative_orientations,\
            histogram_orientations[1]
    else:
        return histogram_orientations[0], relative_orientations, histogram_orientations[1]



@metric("orientations_histogram", "relative_orientations", "orientations_histogram_bins")
def hist_orientations_filtered(absolute_orientations, discrete_positions,
                               maze, orientation_tiles_filter,
                               n_bins=None, max_lim=None, possible_actions=None):
    """
    Compute the histogram of orientations and the relative orientation (when there is motion), but when the tile
    type is in the orientation_tiles_filter list

    This can either produce evenly spaced bins (by specifying n_bins and max_lim) or uneven bins if possible_actions
    are given.

    :param absolute_orientations: list of absolute orientation
    :param discrete_positions: list of discrete positions
    :param maze: maze
    :param orientation_tiles_filter: list of tile types to select
    :param n_bins: number of bins for the histogram
    :param max_lim: orientation corresponding to the last bin
    :param possible_actions: list of possible actions to create the bins
    :return: histogram of orientations, relative orientations, bins for the histogram
    """
    if len(absolute_orientations) != len(discrete_positions):
        raise RuntimeError(f"Wrong input sizes for hist_orientations_filtered {len(absolute_orientations)} != {len(discrete_positions)}")

    bins, min_lim, max_lim = _hist_orientation_commons("hist_orientations_filtered", n_bins, max_lim, possible_actions)

    relative_orientations = []
    for tim in range(0, len(absolute_orientations) - 1):
        if maze.get_tile(discrete_positions[tim]) in orientation_tiles_filter:
            rel_ori = absolute_orientations[tim + 1] - absolute_orientations[tim]
            if rel_ori > max_lim:
                rel_ori = rel_ori - np.pi * 2
            elif rel_ori < min_lim:
                rel_ori = rel_ori - (-np.pi * 2)

            if discrete_positions[tim + 1] != discrete_positions[tim]:
                relative_orientations.append(rel_ori)

    histogram_orientations = np.histogram(relative_orientations, bins=bins, range=(min_lim, max_lim))

    return histogram_orientations[0], relative_orientations, histogram_orientations[1]


@metric("count_moving", "moving_bool_list")
def motion_analysis(discrete_positions):
    """
    Compute the number of times the mouse is moving and a list of booleans indicating whether the mouse has moved
    in a certain step or not

    :param discrete_positions:
    :return: number of motions, boolean list
    """
    count_moving = 0
    moving_bool_list = [False] * len(discrete_positions)

    for i in range(1, len(discrete_positions)):
        if discrete_positions[i] != discrete_positions[i-1]:
            count_moving += 1
            moving_bool_list[i] = True
    return count_moving, moving_bool_list


@metric("histogram_time_moving")
def hist_time_moving(moving_bool_list):
    """
    Compute a histogram of move/not move actions

    Returns a list of actions (n_steps, action_type) where action_type is 1 if moving, 0 if not moving and
    n_steps is the duration of the action.

    :param moving_bool_list: boolean list of moving/not moving
    :return: list of actions
    """
    bin_mov = 1
    time_moving_histogram = []
    current_action = moving_bool_list[0]
    for t in range(1, len(moving_bool_list)):
        if moving_bool_list[t] == current_action:
            bin_mov += 1
        else:
            time_moving_histogram.append((bin_mov, int(current_action)))
            bin_mov = 1
            current_action = not current_action
    time_moving_histogram.append((bin_mov, int(current_action)))
    return time_moving_histogram


@metric("subareas_histogram")
def subareas(discrete_positions, maze):
    """
    Compute the histogram of subareas

    :param discrete_positions: list of discrete positions
    :param maze: maze
    :return: histogram of subareas
    """
    pos_subareas = maze.get_subareas(discrete_positions)
    all_subareas = maze.subareas
    min_sub = min(all_subareas)
    max_sub = max(all_subareas)
    num_subareas = max_sub - min_sub + 1
    hist_subareas = np.histogram(pos_subareas, bins=num_subareas, range=[min_sub, max_sub])
    return hist_subareas[0]


@metric("bouts_mov", "bouts_stop")
def mov_bouts(histogram_time_moving):
    """
    Compute the moving/not-moving over median bouts

    :param histogram_time_moving: list of (n_steps, action_type)
    :return: bouts moving, bouts not moving
    """
    median_period = np.median([b[0] for b in histogram_time_moving])
    mov = 0
    stop = 0
    for b in histogram_time_moving:
        if b[0] > median_period and b[1] == 1:
            mov += b[0]
        elif b[0] > median_period and b[1] == 0:
            stop += b[0]
    return mov, stop


@metric("static_intervals")
def compute_static_intervals(histogram_time_moving):
    """
    Compute the number of static intervals

    A static interval is a sequence of consecutive static bouts

    :param histogram_time_moving: list of (n_steps, action_type)
    :return: number of static intervals
    """
    stop = 0
    for b in histogram_time_moving:
        if b[1] == 0:
            stop += 1
    return stop


def get_endpoints_without_mouse(cont_pos, cont_ori, possible_actions, maze):
    """
    Compute a list of possible endpoints from a given position, with absolute coordinates

    :return: list of np.array of discrete endpoints (2 coordinates)
    """
    # compute rotation matrix (rotation matrix for counterclockwise rotation)
    rot = np.array([[np.cos(cont_ori), -np.sin(cont_ori)], [np.sin(cont_ori), np.cos(cont_ori)]])

    ends = np.zeros((len(possible_actions), 2))
    for idx, act in enumerate(possible_actions):
        ends[idx] = cont_pos + np.matmul(rot, act)
    return maze.cont2disc_list(ends)


@metric("absolute_orientations")
def compute_orientations(continuous_positions, maze=None, possible_actions=None):
    """
    Compute the absolute orientations from a list of positions

    :param continuous_positions: list of continuous positions
    :param maze: maze object
    :param possible_actions: list of tuples describing the next relative possible actions in the model
    :return: list of absolute orientations
    """
    orientations = []
    prev_ori = 0
    theta = None

    for i in range(len(continuous_positions) - 1):
        p2 = continuous_positions[i + 1]
        p1 = continuous_positions[i]

        if maze is None and possible_actions is None:
            condition = not np.allclose(p1, p2)
        else:
            condition = not np.allclose(p1, p2) and maze.cont2disc(p2) in get_endpoints_without_mouse(p1, prev_ori,
                                                                                                      possible_actions,
                                                                                                      maze)
        if condition:
            # get angle in radians
            theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            # correct angle
            if theta > np.pi:
                theta = np.pi * 2 - theta
            elif theta < -np.pi:
                theta = -np.pi * 2 - theta

            orientations.append(theta)
        else:
            if i > 0 and len(orientations) != 0:
                orientations.append(orientations[-1])

        if theta is not None:
            prev_ori = theta

    return [orientations[0]] * (len(continuous_positions) - len(orientations)) + orientations


@metric("it_2_visit_perc_maze")
def it_perc_visited_maze(discrete_positions, maze, percentage=80):
    """
    Record the percentage of explored maze in time

    :param discrete_positions: list of discrete positions
    :param maze: maze
    :param percentage: integer of the wanted percentage of visited maze for the analyses

    :return: (list of the evolution of the percentage of visited maze)
             integer of the number of iterations to cover the wanted percentage of the maze (if reached)
    """
    number_visitable_tiles = maze.get_number_visitable_tiles()
    number_visited_tiles = 0
    visited_tiles = []
    # perc_visit_maze_all_sim = []
    num_it_perc_visit = len(discrete_positions)

    for idx, poss in enumerate(discrete_positions):
        if poss not in visited_tiles:
            number_visited_tiles += 1
            perc_visit_maze = (number_visited_tiles * 100) / number_visitable_tiles
            visited_tiles.append(copy.copy(poss))

            # perc_visit_maze_all_sim.append(copy.copy(perc_visit_maze))

            if perc_visit_maze >= percentage:
                num_it_perc_visit = idx
                break

            if perc_visit_maze > 100:
                raise RuntimeError(f"perc_visit_maze: {perc_visit_maze} > 100 !!!")

    # return perc_visit_maze_all_sim, num_it_perc_visit
    return num_it_perc_visit



@metric("corridors_switches")
def arms(discrete_positions, maze):
    """
    Count the number of switches between the 2 corridors

    :param discrete_positions: list of discrete positions
    :param maze: maze

    :return: integer of the number of switch between the 2 corridors (subareas 1 and 7)
    """
    enter_idx = None
    enter_side = None
    time_diffs = []

    for idx, tile in enumerate(discrete_positions):
        sub_pos = maze.get_subarea(tile[0], tile[1])
        if sub_pos == 1:
            if enter_idx is None:
                enter_idx = idx
                enter_side = 1
            elif enter_side == 7:
                time_diffs.append(idx - enter_idx)
                enter_idx = idx
                enter_side = 1

        elif sub_pos == 7:
            if enter_idx is None:
                enter_idx = idx
                enter_side = 7
            elif enter_side == 1:
                time_diffs.append(idx - enter_idx)
                enter_idx = idx
                enter_side = 7

    return len(time_diffs)

