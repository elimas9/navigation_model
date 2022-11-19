class StandardExperiment:
    """
    Class representing an Experiment

    An Experiment is using a specific exploration model, a given agent ('individual') and a precise maze to perform all
    the steps required to simulate an explorative behaviour
    """

    def __init__(self, exp_model, maze, mouse, debug=False):
        """
        Create the experiment

        :param exp_model: exploration model (BehaviouralModel)
        :param maze: the maze
        :param mouse: the mouse
        :param debug: enable debug prints
        """
        self._exp_model = exp_model
        self._maze = maze
        self._mouse = mouse
        self._debug = debug

    def run(self, iterations):
        """
        Run the experiment

        :param iterations: number of steps to simulate
        """
        if self._debug:
            print("*** experiment start")

        self._mouse.enable_history(True)  # we want to save the positions and orientation
        for it in range(iterations):
            # prepare arguments
            # this part sucks a bit and we should use a mechanism that allows each component to specify what it needs
            next_positions_cont = self._mouse.get_endpoints()
            is_visitable = self._maze.are_movements_possible_cont(self._mouse.position, next_positions_cont)
            next_positions_cont = next_positions_cont[is_visitable]
            next_positions = self._maze.cont2disc_list(next_positions_cont)
            next_tiles = self._maze.get_tiles(next_positions)
            x, y = self._maze.cont2disc(self._mouse.position)

            # decision making
            new_position = self._exp_model.decision_making(next_positions=next_positions,
                                                           next_tiles=next_tiles,
                                                           x=x, y=y,
                                                           is_visitable=is_visitable,
                                                           next_positions_cont=next_positions_cont)

            # move agent
            self._mouse.move_to(new_position)

    def plot(self, ax, maze_tiles=True, maze_borders=False, maze_grid=False,
             mouse_color="y", mouse_endpoints=False, mouse_trajectory_color="y", mouse_size=0.1):
        """
        Plot the current status of the experiment, including the maze and the mouse

        :param ax: axis where to plot
        :param maze_tiles: if True plot colored tiles
        :param maze_borders: if True add maze borders
        :param maze_grid: if True add grid
        :param mouse_color: color of the mouse
        :param mouse_endpoints: if True plot the current mouse endpoints
        :param mouse_trajectory_color: color of the mouse trajectory
        :param mouse_size: size of the mouse (same units as the maze tile size)
        """
        self._maze.plot(ax, tiles=maze_tiles, borders=maze_borders, grid=maze_grid)
        self._mouse.plot_history(ax, c=mouse_trajectory_color)
        self._mouse.plot(ax, c=mouse_color, endpoints=mouse_endpoints, size=mouse_size)
