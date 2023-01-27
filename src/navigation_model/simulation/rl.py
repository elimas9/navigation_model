import sys

import numpy as np


class ConditioningExperiment:
    """
    Class representing a conditioning experiment

    In the conditioning experiment, the mouse follows a specific trajectory and receives a reward at each step.
    Reinforcement learning is used to learn a value function for the states of the maze.
    Replays can be performed after one or multiple runs.
    """

    def __init__(self, maze, algorithm, replay_strategy=None, v_table=None, debug=False):
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
            self._alg.v_table = v_table
        else:
            self._alg.v_table = np.zeros(self._maze.shape)

        self._replay_strategy = replay_strategy

        self._debug = debug

    def run(self, trajectory, reward, do_replays=False):
        """
        Run the experiment for a certain trajectory and with a certain reward

        :param trajectory: predetermined path to follow (list of continuous positions)
        :param reward: reward obtained at each step (list of rewards)
        :param do_replays: if True, perform offline replay after the run
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

            to_buffer = self._alg.update_and_to_buffer(cs, s, cs1, s1, r)
            self._replay_strategy.append((s, to_buffer, r))

            cs = cs1
            s = s1

        if do_replays:
            if self._replay_strategy is None:
                print("Asked to do replay, but no replay strategy given", file=sys.stderr)
            else:
                self._replay_strategy.offline_replays(self._alg)

    def do_replays(self):
        """
        Perform replays depending on the replay type
        """
        self._replay_strategy.offline_replays(self._alg)

    @property
    def v_table(self):
        return self._alg.v_table


############
# algorithms
class RLAlgorithm:

    def __init__(self):
        self._v_table = None

    @property
    def v_table(self):
        if self._v_table is None:
            raise RuntimeError("V-table not initialized")
        return self._v_table

    @v_table.setter
    def v_table(self, v_table):
        self._v_table = v_table

    def update_v_table(self, s, dv):
        self._v_table[s[0], s[1]] += dv

    def update_and_to_buffer(self, cs, s, cs1, s1, r):
        """
        Updates V(s) based on the current transition and returns a value to put in the replay buffer

        :param cs: continuous starting position
        :param s: starting state
        :param cs1: continuous arrival position
        :param s1: arrival state
        :param r: reward
        :return: value to store in the buffer
        """
        raise NotImplementedError()

    def update(self, s, s1, r):
        """
        Updates V(s) based on the current transition

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
        super().__init__()
        self._alpha = alpha
        self._discount_rate = discount_rate
        self._mouse = mouse
        self._maze = maze

    def update_and_to_buffer(self, cs, s, cs1, s1, r):
        self._mouse.move_to(cs)
        endpoints = self._mouse.get_endpoints()
        next_states = self._maze.cont2disc_list(endpoints)
        are_visitable = self._maze.are_visitable(next_states)
        next_states = [ns for i, ns in enumerate(next_states) if are_visitable[i]]

        self.update(s, next_states, r)

        self._mouse.move_to(cs1)

        return next_states

    def update(self, s, s1, r):
        dv = self._discount_rate * np.max([self._v_table[es[0], es[1]] for es in s1])
        dv = self._alpha * (r + dv - self._v_table[s[0], s[1]])
        self.update_v_table(s, dv)
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
        super().__init__()
        self._alpha = alpha
        self._discount_rate = discount_rate
        self._mouse = mouse
        self._maze = maze

    def update_and_to_buffer(self, cs, s, cs1, s1, r):
        self._mouse.move_to(cs)
        endpoints = self._mouse.get_endpoints()
        next_states = self._maze.cont2disc_list(endpoints)
        are_visitable = self._maze.are_visitable(next_states)
        next_states = [ns for i, ns in enumerate(next_states) if are_visitable[i]]

        self.update(s, next_states, r)

        self._mouse.move_to(cs1)

        return next_states

    def update(self, s, s1, r):
        dv = self._discount_rate * np.min([self._v_table[es[0], es[1]] for es in s1])
        dv = self._alpha * (r + dv - self._v_table[s[0], s[1]])
        self.update_v_table(s, dv)
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
        super().__init__()
        self._alpha = alpha
        self._discount_rate = discount_rate

    def update_and_to_buffer(self, cs, s, cs1, s1, r):
        self.update(s, s1, r)
        return s1

    def update(self, s, s1, r):
        dv = self._alpha * (r + self._discount_rate * self._v_table[s1[0], s1[1]] - self._v_table[s[0], s[1]])
        self.update_v_table(s, dv)
        return dv


###################
# replay strategies
class ReplayStrategy:
    """
    Base class for replay strategies
    """

    def append(self, update):
        """
        Add an update transition to the buffer
        """
        raise NotImplementedError

    def clear_buffer(self):
        """
        Reset the replay buffer
        """
        raise NotImplementedError

    def offline_replays(self, alg):
        """
        Do replays with the current strategy
        """
        raise NotImplementedError

class ShuffledReplays(ReplayStrategy):
    """
    Shuffle the replay buffer
    """
    def __init__(self, n_times, seed=None):
        """
        Create a shuffled replay strategy

        :param n_times: number of times the full sequence is presented
        :param seed: seed
        """
        super().__init__()

        self._n_times = n_times
        self._buffer = []
        self._rng = np.random.default_rng(seed)

    def append(self, update):
        self._buffer.append(update)

    def clear_buffer(self):
        self._buffer = []

    def offline_replays(self, alg):
        """
        Do replays with the current strategy
        """
        for _ in range(self._n_times):
            self._rng.shuffle(self._buffer)
            for s, s1, r in self._buffer:
                alg.update(s, s1, r)
