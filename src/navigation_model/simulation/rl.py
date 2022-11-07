import numpy as np


class ConditioningExperiment:
    """
    Class representing a conditioning experiment

    In the conditioning experiment, the mouse follows a specific trajectory and receives a reward at each step.
    Reinforcement learning is used to learn a value function for the states of the maze.
    Replays can be performed after one or multiple runs.
    """

    def __init__(self, maze, algorithm, replay_strategy, v_table=None, debug=False):
        """
        Create the replay experiment

        If v_table is used then the learning is "continuning" from that table.

        :param maze: the maze
        :param algorithm: RL algorithm to use
        :param replay_strategy: replay strategy to use
        :param v_table: starting value table
        :param debug: enable debug prints
        """
        self._maze = maze

        self._alg = algorithm

        if v_table is not None:
            self._v_table = v_table
        else:
            self._v_table = np.zeros(self._maze.shape)

        self._replay_strategy = replay_strategy
        self._replay_buffer = []

        self._debug = debug

    def run(self, trajectory, reward, do_replays=0):
        """
        Run the experiment for a certain trajectory and with a certain reward

        :param trajectory: predetermined path to follow (list of continuous positions)
        :param reward: reward obtained at each step (list of rewards)
        :param do_replays: number of replay sequences
        """
        # checks
        if len(trajectory) != len(reward):
            raise RuntimeError("trajectory and reward must have the same length")

        cs = trajectory[0]
        s = self._maze.get_closest_visitable_cont(cs)
        for i in range(1, len(trajectory)):
            cs1 = trajectory[i]
            s1 = self._maze.get_closest_visitable_cont(cs1)
            r = reward[i]

            dv, to_buffer = self._alg.get_update_and_to_buffer(self._v_table, cs, s, cs1, s1, r)
            self._v_table[s[0], s[1]] += dv
            self._replay_buffer.append((s, to_buffer, r))

            cs = cs1
            s = s1

        self.do_replays(n_times=do_replays)

    def do_replays(self, n_times):
        """
        Perform replays depending on the replay type

        :param n_times: number of times the whole sequence is replayed
        """
        for _ in range(n_times):
            buffer = self._replay_strategy(self._replay_buffer)
            for s, s1, r in buffer:
                dv = self._alg.get_update(self._v_table, s, s1, r)
                self._v_table[s[0], s[1]] += dv

    def reset_replay_buffer(self):
        """
        Reset the replay buffer
        """
        self._replay_buffer = []

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @property
    def v_table(self):
        return self._v_table


############
# algorithms
class RLAlgorithm:

    def get_update_and_to_buffer(self, v_table, cs, s, cs1, s1, r):
        """
        Returns the computed update to V(s) and the value to store in the buffer for replays

        The returned update should be added to V(s).

        :param v_table: current V table
        :param cs: continuous starting position
        :param s: starting state
        :param cs1: continuous arrival position
        :param s1: arrival state
        :param r: reward
        :return: update to V(s), value to store in the buffer
        """
        raise NotImplementedError()

    def get_update(self, v_table, s, s1, r):
        """
        Returns the computed update to V(s)

        The returned update should be added to V(s).

        :param v_table: current V table
        :param s: starting state
        :param s1: list of arrival states
        :param r: reward
        :return: update to V(s)
        """
        raise NotImplementedError()


class PositiveConditioning(RLAlgorithm):
    """
    Positive conditioning learning

    V(s) = V(s) + alpha [R + gamma max(V(s')) - V(s)]
    """

    def __init__(self, alpha, discount_rate, mouse, maze):
        """
        The mouse and maze are needed to compute the next possible states

        :param alpha: learning rate
        :param discount_rate: discount rate
        :param mouse: the mouse
        :param maze: the maze
        """
        self._alpha = alpha
        self._discount_rate = discount_rate
        self._mouse = mouse
        self._maze = maze

    def get_update_and_to_buffer(self, v_table, cs, s, cs1, s1, r):
        self._mouse.move_to(cs)
        endpoints = self._mouse.get_endpoints()
        next_states = self._maze.cont2disc_list(endpoints)
        are_visitable = self._maze.are_visitable(next_states)
        next_states = [ns for i, ns in enumerate(next_states) if are_visitable[i]]

        dv = self.get_update(v_table, s, next_states, r)

        self._mouse.move_to(cs1)

        return dv, next_states

    def get_update(self, v_table, s, s1, r):
        dv = self._discount_rate * np.max([v_table[es[0], es[1]] for es in s1])
        dv = self._alpha * (r + dv - v_table[s[0], s[1]])
        return dv


class NegativeConditioning(RLAlgorithm):
    """
    Positive conditioning learning

    V(s) = V(s) + alpha [P + gamma min(V(s')) - V(s)]
    """

    def __init__(self, alpha, discount_rate, mouse, maze):
        """
        The mouse and maze are needed to compute the next possible states

        :param alpha: learning rate
        :param discount_rate: discount rate
        :param mouse: the mouse
        :param maze: the maze
        """
        self._alpha = alpha
        self._discount_rate = discount_rate
        self._mouse = mouse
        self._maze = maze

    def get_update_and_to_buffer(self, v_table, cs, s, cs1, s1, r):
        self._mouse.move_to(cs)
        endpoints = self._mouse.get_endpoints()
        next_states = self._maze.cont2disc_list(endpoints)
        are_visitable = self._maze.are_visitable(next_states)
        next_states = [ns for i, ns in enumerate(next_states) if are_visitable[i]]

        dv = self.get_update(v_table, s, next_states, r)

        self._mouse.move_to(cs1)

        return dv, next_states

    def get_update(self, v_table, s, s1, r):
        dv = self._discount_rate * np.min([v_table[es[0], es[1]] for es in s1])
        dv = self._alpha * (r + dv - v_table[s[0], s[1]])
        return dv


class TD0(RLAlgorithm):
    """
    TD(0) algorithm

    V(s) = V(s) + alpha [R + gamma V(s') - V(s)]
    """

    def __init__(self, alpha, discount_rate):
        """
        :param alpha:
        :param discount_rate:
        """
        self._alpha = alpha
        self._discount_rate = discount_rate

    def get_update_and_to_buffer(self, v_table, cs, s, cs1, s1, r):
        dv = self.get_update(v_table, s, s1, r)
        return dv, s1

    def get_update(self, v_table, s, s1, r):
        return self._alpha * (r + self._discount_rate * v_table[s1[0], s1[1]] - v_table[s[0], s[1]])


###################
# replay strategies
class ShuffledReplays:
    """
    Shuffle the replay buffer

    This method modifies the original buffer!
    """
    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def __call__(self, buffer):
        self._rng.shuffle(buffer)
        return buffer
