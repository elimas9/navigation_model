import numpy as np
import copy
import matplotlib


def plot_trajectory(ax, trajectory, c="b"):
    """
    Plot a trajectory

    :param ax: axis where to plot
    :param trajectory: list of continuous coordinates
    :param c: color
    :return: list of Line2D
    """
    trajectory = np.array(trajectory)
    return ax.plot(trajectory[:, 0], trajectory[:, 1], c=c)


def plot_reward_sequence(ax, reward_values, reward_positions, marker="x", c="r"):
    """
    Plot events from a sequence of numbers

    An event is plotted if not 0.

    :param ax: axis where to plot
    :param reward_values: list of reward values
    :param reward_positions: list of reward positions (continuous coordinates)
    :param marker: marker
    :param c: color
    :return: PathCollection object
    """
    reward_events = reward_positions[reward_values != 0]
    return plot_reward_events(ax, reward_events, marker, c)


def plot_reward_events(ax, reward_events, marker="x", c="r"):
    """
    Plot a list of reward events

    :param ax: axis where to plot
    :param reward_events: list of coordinates where the reward happened
    :param marker: marker
    :param c: color
    :return: PathCollection
    """
    reward_events = np.array(reward_events)
    return ax.scatter(reward_events[:, 0], reward_events[:, 1], c=c, marker=marker, zorder=3)


def plot_map(ax, map_data, maze=None, cmap=None, norm=None):
    """
    Plot a map (V table, occupancy map, etc)

    If a maze is passed, it will be used to remove non-visitable tiles from the plot.

    :param ax: axis where to plot
    :param map_data: map
    :param maze: optional maze
    :param cmap: color map
    :param norm: normalization
    :return: AxesImage object
    """
    if cmap is None:
        cmap = matplotlib.cm.get_cmap()
    if maze is None:
        data = map_data
    else:
        data = copy.deepcopy(map_data)
        bad_coords = maze.get_non_visitable_coordinates()
        for x, y in bad_coords:
            data[x, y] = np.nan
        cmap.set_bad(color='white')
    # remove border
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax.imshow(data.transpose(), origin="lower", cmap=cmap, norm=norm)


def plot_multiple_maps(fig, maps, rows, columns, maze=None, cmap=None, subplots_ratio=20):
    """
    Plot multiple maps (V table, occupancy map, etc) with a global colorbar

    If a maze is passed, it will be used to remove non-visitable tiles from the plots.

    :param fig: figure where to plot
    :param maps: list of maps
    :param rows: rows for the arrangement
    :param columns: columns for the arrangment
    :param maze: optional maze
    :param cmap: colormap
    :param subplots_ratio: ratio of subplots over colorbar
    :return: list of axes
    """
    if rows * columns < len(maps):
        raise RuntimeError("Not enough rows and columns specified!")

    vmax = np.max(maps)
    vmin = np.min(maps)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    main_gs = matplotlib.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[subplots_ratio, 1])
    cax = fig.add_subplot(main_gs[1])
    v_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(rows, columns, subplot_spec=main_gs[0])

    axs = []
    if rows > 1 and columns > 1:
        for r in range(rows):
            row_axs = []
            for c in range(columns):
                ax = fig.add_subplot(v_grid[r, c])
                idx = r * columns + c
                if idx > len(maps):
                    break
                v = maps[idx]
                plot_map(ax, v, maze=maze, cmap=cmap, norm=norm)
                row_axs.append(ax)
            axs.append(row_axs)
    else:
        for i, v in enumerate(maps):
            ax = fig.add_subplot(v_grid[i])
            plot_map(ax, v, maze=maze, cmap=cmap, norm=norm)

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax)

    return axs
