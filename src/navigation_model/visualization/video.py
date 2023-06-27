import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import progressbar


def create_video(trajectory, maze, delta_t, filepath,
                 reward=None, reward_duration=0.1, reward_position=(0.2, 0.2), reward_size=0.1, reward_color='yellow',
                 trajectory_color="b",
                 fps=60, bitrate=1800,
                 speedup=None, speedup_position=None):
    """
    Create and save a video of a session

    :param trajectory: mouse trajectory
    :param maze: maze
    :param delta_t: float (s) or list of times
    :param filepath: where to save the file
    :param reward: reward sequence (list of 0, 1, -1)
    :param reward_duration: how long to show each reward (s)
    :param reward_position: position of the reward circle
    :param reward_size: size of the reward circle
    :param reward_color: color of the reward circle
    :param trajectory_color: color of the trajectory
    :param fps: frames per second in the output video (if high, it increases the fluency of the video)
    :param bitrate: bitrate of the output video (if high, it increases the quality of the video)
    :param speedup: speedup of the video
    :param speedup_position: position of the speedup label (if None, it will be computed automatically)
    """

    fig = plt.figure()
    ax = fig.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor((0, 0, 0, 0))

    # accept both a list of times and a timestep size
    if type(delta_t) is float:
        max_t = len(trajectory) * delta_t
        delta_t = np.arange(0, max_t, delta_t)
        assert len(delta_t) == len(trajectory)
    else:
        max_t = trajectory[-1]
    all_x = np.array([d[0] for d in trajectory])
    all_y = np.array([d[1] for d in trajectory])

    # plot maze outline
    maze.plot(ax, tiles=False, borders=True)

    # write speedup if given
    if speedup is not None:
        if speedup_position is None:
            speedup_position = (np.max(all_x) * 1.1, 0.0)
        ax.text(*speedup_position, f"{speedup}X", fontsize=14, transform=ax.transAxes)

    # create empty trajectory line
    ln = ax.plot([], [], c=trajectory_color)[0]

    # create empty trajectory line for the final positions
    ln_end = ax.plot([], [], c="blue")[0]

    # create empty tuple for the final position
    # pnt_end = plt.Circle((), reward_size, color="red")#, alpha=0)
    # ax.add_artist(pnt_end)
    ln_pnt_end = ax.plot([], [], c="red", marker='D', markersize=5)[0]


    # create circle to show reward
    if reward is not None:
        circle = plt.Circle(reward_position, reward_size, color=reward_color, alpha=0)
        ax.add_artist(circle)

    # blitting (might be an useless optimization)
    def init():
        if reward is not None:
            return ln, circle, ln_end, ln_pnt_end
        else:
            return ln, ln_end, ln_pnt_end,

    # actual plotting function
    max_time = len(trajectory) - 1
    bar = progressbar.ProgressBar(max_value=max_time)
    last_reward_t = -reward_duration
    last_dt = 0

    def update(dt):
        nonlocal last_dt, last_reward_t
        # x = all_x[:dt-5]
        # y = all_y[:dt-5]
        x = all_x[:max(dt-5, 0)]
        y = all_y[:max(dt-5, 0)]

        # x_end = all_x[dt-6:dt]
        # y_end = all_y[dt-6:dt]
        x_end = all_x[max(dt-6, 0):dt]
        y_end = all_y[max(dt-6, 0):dt]

        # x_end_pnt = all_x[dt-1]
        # y_end_pnt = all_y[dt-1]
        x_end_pnt = all_x[max(dt-1, 0)]
        y_end_pnt = all_y[max(dt-1, 0)]

        ln.set_data(x, y)
        ln_end.set_data(x_end, y_end)
        ln_pnt_end.set_data(x_end_pnt, y_end_pnt)
        # pnt_end.set_data((x_end_pnt, y_end_pnt))
        # pnt_end = plt.Circle((x_end_pnt, y_end_pnt), reward_size, color="red")  # , alpha=0)

        # update last reward time
        if reward is not None:
            for i in range(last_dt, dt):
                if reward[i] != 0:
                    last_reward_t = delta_t[i]
            t = delta_t[dt]
            if last_reward_t + reward_duration > t:
                circle.set_alpha(1)
            else:
                circle.set_alpha(0)

            bar.update(dt)
            last_dt = dt
            return ln, circle, ln_end, ln_pnt_end
        else:
            bar.update(dt)
            last_dt = dt
            return ln, ln_end, ln_pnt_end,

    # create animation
    if speedup is None:
        speedup = 1
    n_frames = int(max_t * fps / speedup)
    ani = FuncAnimation(fig,
                        update,
                        frames=np.linspace(0, max_time, num=n_frames, dtype=int),
                        init_func=init,
                        blit=True)

    # save to file
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    print("Saving video:")
    ani.save(filepath, writer=writer)
    bar.update(max_time)
