import numpy as np
import copy
import matplotlib
import seaborn as sns


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
    reward_values = np.array(reward_values)
    reward_positions = np.array(reward_positions)
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


def plot_map(ax, map_data, maze=None, cmap=None, norm=None, title=None):
    """
    Plot a map (V table, occupancy map, etc)

    If a maze is passed, it will be used to remove non-visitable tiles from the plot.

    :param ax: axis where to plot
    :param map_data: map
    :param maze: optional maze
    :param cmap: color map
    :param norm: normalization
    :param title: axis title
    :return: AxesImage object
    """
    if cmap is None:
        cmap = matplotlib.cm.get_cmap()
    if maze is None:
        data = map_data
        extent = None
    else:
        data = copy.deepcopy(map_data)
        bad_coords = maze.get_non_visitable_coordinates()
        for x, y in bad_coords:
            data[x, y] = np.nan
        cmap.set_bad(color='white')
        extent = [0, maze.size_x * maze.size_tile,
                  0, maze.size_y * maze.size_tile]
    # remove border
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return ax.imshow(data.transpose(), origin="lower", cmap=cmap, norm=norm, extent=extent)


def plot_multiple_maps(fig, maps, rows, columns, maze=None, cmap=None, subplots_ratio=20, titles=None):
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
    :param titles: set titles to maps
    :return: np.arry of axes (shape=rows x columns)
    """
    if rows * columns < len(maps):
        raise RuntimeError("Not enough rows and columns specified!")

    if titles is None:
        titles = [None] * len(maps)
    elif len(titles) < len(maps):
        titles = titles + [None] * (len(maps) - len(titles))

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
                idx = r * columns + c
                if idx >= len(maps):
                    row_axs.append(None)
                else:
                    ax = fig.add_subplot(v_grid[r, c])
                    v = maps[idx]
                    plot_map(ax, v, maze=maze, cmap=cmap, norm=norm, title=titles[idx])
                    row_axs.append(ax)
            axs.append(row_axs)
    else:
        for i, v in enumerate(maps):
            ax = fig.add_subplot(v_grid[i])
            plot_map(ax, v, maze=maze, cmap=cmap, norm=norm, title=titles[i])
            axs.append(ax)

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax)

    return np.array(axs)


def plot_boxes(ax, data, labels=None, title=None, show_points=True, box_args=None, strip_args=None):
    """
    Box and whisker plot with optional jittered points

    The input can be given either as a dictionary (without specifying labels), as in

    data={"category1": category1_data,
          "category2": category2_data,
          "category3": category3_data}

    or as a list of vectors with optional labels

    data=[category1_data, category2_data, category3_data], labels=["category1", "category2", "category3"]

    :param ax: axis where to plot
    :param data: dictionary or list with data
    :param labels: optional list of labels
    :param title: axis title
    :param show_points: if True show jittered points
    :param box_args: dictionary of parameters passed to sns.boxplot
    :param strip_args: dictionary of parameters passed to sns.stripplot (if enabled)
    """
    if type(data) is dict:
        labels = data.keys()
        data = np.array([d for d in data.values()]).transpose()
    else:
        data = np.array([d for d in data]).transpose()
    data_length = data.shape[-1]

    # we need to do a different thing when plotting single points
    if len(data.shape) == 1:
        # and we plot just the means
        data = [np.array([d]) for d in data]
        sns.boxplot(data=data, ax=ax)
        data_length = len(data)
    else:
        # actual plot
        if box_args is None:
            box_args = {}
        sns.boxplot(data=data, ax=ax, **box_args)

        if show_points:
            if strip_args is None:
                strip_args = {}
            sns.stripplot(data=data, ax=ax, **strip_args)

    # labels
    if labels is not None:
        if data_length > len(labels):
            raise RuntimeError("Not enough labels for box plots")

        ax.set_xticklabels(labels)

    if title is not None:
        ax.set_title(title)


def plot_multiple_boxes(fig, data, rows, columns, labels=None, titles=None, show_points=True,
                        box_args=None, strip_args=None):
    """
    Multiple box and whisker plot in the same figure

    :param fig: figure where to plot
    :param data: list of dictionaries of lists (see plot_boxes)
    :param rows: rows for the arrangement
    :param columns: columns for the arrangment
    :param labels: optional list of list of labels
    :param titles: list of titles for the axes
    :param show_points: if True show jittered points
    :param box_args: dictionary of parameters passed to sns.boxplot
    :param strip_args: dictionary of parameters passed to sns.stripplot (if enabled)
    :return: np.arry of axes (shape=rows x columns)
    """
    if rows * columns < len(data):
        raise RuntimeError("Not enough rows and columns specified!")

    # adjust titles
    if titles is None:
        titles = [None] * len(data)
    elif len(titles) < len(data):
        titles = titles + [None] * (len(data) - len(titles))

    # adjust labels so that the check is deferred to plot_boxes
    if labels is None:
        labels = [None] * len(data)

    axs = fig.subplots(rows, columns, sharey=True)

    if rows > 1 and columns > 1:
        for r in range(rows):
            for c in range(columns):
                idx = r * columns + c
                if idx >= len(data):
                    fig.delaxes(axs[r, c])
                else:
                    plot_boxes(axs[r, c], data[idx], labels=labels[idx], title=titles[idx],
                               show_points=show_points, box_args=box_args, strip_args=strip_args)
    else:
        for i, v in enumerate(data):
            plot_boxes(axs[i], data[i], labels=labels[i], title=titles[i],
                       show_points=show_points, box_args=box_args, strip_args=strip_args)
    return axs
