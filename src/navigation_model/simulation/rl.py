import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np


############
# types
@dataclass
class Transition:
    """
    Class representing a transition
    """
    cs: Any = None                             # starting continuous position
    ori: Any = None                            # starting orientation
    s: Any = None                              # starting discrete position
    cs1: Any = None                            # arriving continuous position
    ori1: Any = None                           # arriving orientation
    s1: Any = None                             # arriving discrete state
    r: Any = None                              # reward
    extra: dict = field(default_factory=dict)  # dictionary that can be used by different algorithms for extra info


# update type
@dataclass
class Update:
    """
    Class representing a value update caused by a transition
    """
    transition: Transition = None  # transition that caused the update
    dv: Any = None                 # value update
    rpe: Any = None                # reward prediction error (absolute)


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

        If v_table is used then the learning is "continuing" from that table.

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

    def run(self, trajectory, orientations, reward, do_replays=False):
        """
        Run the experiment for a certain trajectory and with a certain reward

        :param trajectory: predetermined path to follow (list of continuous positions)
        :param orientations: orientations of the mouse in the path (list of continuous orientations)
        :param reward: reward obtained at each step (list of rewards)
        :param do_replays: if True, perform offline replay after the run
        """
        # checks
        if len(trajectory) != len(reward):
            raise RuntimeError("trajectory and reward must have the same length")

        cs = trajectory[0]
        ori = orientations[0]
        s = self._maze.get_closest_visitable_cont(cs)
        for i in range(1, len(trajectory)):
            t = Transition(cs=cs,
                           ori=ori,
                           s=s,
                           cs1=trajectory[i],
                           ori1=orientations[i],
                           s1=self._maze.get_closest_visitable_cont(trajectory[i]),
                           r=reward[i])

            update = self._alg.update(t)
            self._replay_strategy.append(update)

            cs = t.cs1
            ori = t.ori1
            s = t.s1

        if do_replays:
            if self._replay_strategy is None:
                print("Asked to do replay, but no replay strategy given", file=sys.stderr)
            else:
                self._replay_strategy.offline_replays(self._alg)

    def do_replays(self):
        """
        Perform replays depending on the replay type
        """
        if self._replay_strategy is None:
            print("Asked to do replay, but no replay strategy given", file=sys.stderr)
        else:
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

    def update(self, t):
        """
        Updates V(s) based on the current transition and returns a value to put in the replay buffer

        :param t: Transition
        :return: Update to store in the buffer
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

    def update(self, t):
        if "next_states" not in t.extra:
            self._mouse.move_to(t.cs)
            endpoints = self._mouse.get_endpoints()
            next_states = self._maze.cont2disc_list(endpoints)
            are_visitable = self._maze.are_visitable(next_states)
            t.extra["next_states"] = [ns for i, ns in enumerate(next_states) if are_visitable[i]]

        dv = self._discount_rate * np.max([self._v_table[es[0], es[1]] for es in t.extra["next_states"]])
        rpe = t.r + dv - self._v_table[t.s[0], t.s[1]]
        dv = self._alpha * rpe
        self.update_v_table(t.s, dv)

        self._mouse.move_to(t.cs1)

        # create update
        return Update(transition=t,
                      dv=dv,
                      rpe=abs(rpe))


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

    def update(self, t):
        if "next_states" not in t.extra:
            self._mouse.move_to(t.cs)
            endpoints = self._mouse.get_endpoints()
            next_states = self._maze.cont2disc_list(endpoints)
            are_visitable = self._maze.are_visitable(next_states)
            t.extra["next_states"] = [ns for i, ns in enumerate(next_states) if are_visitable[i]]

        dv = self._discount_rate * np.min([self._v_table[es[0], es[1]] for es in t.extra["next_states"]])
        rpe = t.r + dv - self._v_table[t.s[0], t.s[1]]
        dv = self._alpha * rpe
        self.update_v_table(t.s, dv)

        self._mouse.move_to(t.cs1)

        # create update
        return Update(transition=t,
                      dv=dv,
                      rpe=abs(rpe))


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

    def update(self, t):
        rpe = t.r + self._discount_rate * self._v_table[t.s1[0], t.s1[1]] - self._v_table[t.s[0], t.s[1]]
        dv = self._alpha * rpe
        self.update_v_table(t.s, dv)
        return Update(transition=t,
                      dv=dv,
                      rpe=abs(rpe))


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
        for _ in range(self._n_times):
            self._rng.shuffle(self._buffer)
            for update in self._buffer:
                alg.update(update.transition)
