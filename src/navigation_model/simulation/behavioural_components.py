from enum import IntEnum

import numpy as np

from scipy.stats import vonmises

from navigation_model.simulation.maze import StandardTiles


class ParameterDescriptor:
    """
    Class representing a descriptor for a parameter

    A descriptor has a name, minimun and maximum values and the default value, but not the actual value.
    This way, it can be directly added to the class without sharing the value among instances.
    """

    def __init__(self, name, minv, maxv, default):
        """
        Creates a new descriptor

        :param name: name of the descriptor
        :param minv: minimum default value
        :param maxv: maximum default value
        :param default: default value
        """
        self._name = name
        self._minv = minv
        self._maxv = maxv
        self._default = default

    @property
    def name(self):
        """
        Name of the descriptor

        :return: name
        """
        return self._name

    @property
    def minv(self):
        """
        Minimum default value of the descriptor

        :return: minimum default value
        """
        return self._minv

    @property
    def maxv(self):
        """
        Maximum default value of the descriptor

        :return: maximum default value
        """
        return self._maxv

    @property
    def default(self):
        """
        Default value of the descriptor

        :return: default value
        """
        return self._default

    def __str__(self):
        return f"[{self.name}; range: ({self.minv}; {self.maxv}); default: {self.default}]"


class Parameter:
    """
    Class representing a parameter of a component

    A parameter can be created from a descriptor, but it also holds the current value. Thus, it is dynamically
    created upon component creation.
    """

    def __init__(self, descriptor, **kwargs):
        """
        Creates a parameter

        :param descriptor: descriptor of the parameter
        :param kwargs: optional value for the parameter
        """
        self._descriptor = descriptor
        self.val = kwargs.get(self.name, self._descriptor.default)

    @property
    def name(self):
        """
        Name of the parameter

        :return: name
        """
        return self._descriptor.name

    @property
    def minv(self):
        """
        Minimum value of the parameter

        :return: minimum value
        """
        return self._descriptor.minv

    @property
    def maxv(self):
        """
        Maximum value of the parameter

        :return: maximum value
        """
        return self._descriptor.maxv

    @property
    def val(self):
        """
        Current value of the parameter

        :return: value
        """
        return self._val

    @val.setter
    def val(self, value):
        """
        Setter for the value of the parameter

        :param value: new value
        """
        if not self.minv <= value <= self.maxv:
            raise RuntimeError(
                f"Wrong value set for property '{self.name}' ({value}), must be in range [{self.minv}; {self.maxv}]")
        self._val = value

    def __str__(self):
        return f"[{self.name} = {self.val}; range: ({self.minv}; {self.maxv})]"


def parameter(name, doc, minv=-np.inf, maxv=np.inf, default=0.0, isweight=False):
    """
    Decorator to add a parameter to a component

    :param name: name of the parameter
    :param doc: docstring
    :param minv: minimum value of the parameter
    :param maxv: maximum value of the parameter
    :param default: default value of the parameter
    :param isweight: True if the component is a weight, False otherwise
    """

    def deco(cls):

        # check that the class is component
        if not issubclass(cls, Component):
            raise RuntimeError("Trying to apply a parameter decorator to a class not inheriting from Component")

        if isweight and hasattr(cls, "_weight_idx"):
            raise RuntimeError("A component cannot have more than one weight")

        par = ParameterDescriptor(name, minv, maxv, default)

        existing_doc = "" if cls.__doc__ is None else cls.__doc__
        if hasattr(cls, "_descriptors"):

            # check for duplicates
            all_pars = getattr(cls, "_descriptors")
            all_names = [p.name for p in all_pars]
            if name in all_names:
                raise RuntimeError(f"Component cannot have the same parameter {name} twice")
            all_pars.append(par)

            # update doc
            cls.__doc__ = existing_doc + "\n        " + name + ": " + doc

        else:

            # add list of parameters
            setattr(cls, "_descriptors", [par])

            # update doc
            cls.__doc__ = existing_doc + "\n\n    Parameters for this component:\n        " + name + ": " + doc

        # add properties
        last_par_idx = len(cls._descriptors) - 1

        def get_attr(self):
            return self._parameters[last_par_idx].val

        def set_attr(self, value):
            self._parameters[last_par_idx].val = value

        prop = property(get_attr, set_attr, doc=doc)
        setattr(cls, name, prop)

        if isweight:
            setattr(cls, "_weight_idx", last_par_idx)

        return cls
    return deco


def weight(name, doc="weight of the component", minv=0.0, maxv=10.0, default=1.0):
    """
    Decorator to add the weight to a component

    :param name: name of the parameter
    :param doc: docstring
    :param minv: minimum default value of the parameter
    :param maxv: maximum default value of the parameter
    :param default: default value of the parameter
    """
    return parameter(name, doc=doc, minv=minv, maxv=maxv, default=default, isweight=True)


# noinspection PyUnresolvedReferences
# noinspection PyProtectedMember
class Component:
    """
    Generic interface for a component

    A component is capable of assigning values to states. These values are used to perform a decision making process.
    """

    def __init__(self, debug=False, active=True, **kwargs):
        """
        Create a Component.

        :param debug: boolean to set the debug
        :param active: is false, output value of the component will be 0
        :param kwargs:
        """
        self._val = 0

        self.active = active

        self.debug = debug

        if "_parameters" in kwargs:
            # check length and copy parameters
            self._parameters = kwargs["_parameters"]
            if len(self._parameters) != len(type(self)._descriptors):
                raise RuntimeError(f"Wrong number of parameters ({len(self._parameters)}). "
                                   f"Required: {len(type(self)._descriptors)}.")
        else:
            # build the paramters from descriptors
            self._parameters = []
            for pd in type(self)._descriptors:
                try:
                    self._parameters.append(Parameter(pd, **kwargs))
                except KeyError:
                    raise RuntimeError(f"Value for {pd.name} has not been specified")

    @property
    def weight(self):
        """
        Float describing the weight for the component in the model

        :return: weight
        """
        return self._parameters[type(self)._weight_idx].val

    @property
    def values(self):
        """
        Non-weighted current values of the component

        :return: non-weighted current values of the component
        """
        if self.active:
            return self._val
        return 0

    @property
    def weighted_values(self):
        """
        Weighted current values of the component

        :return: weighted current values of the component
        """
        if self.active:
            return self._val * self.weight
        return 0

    @property
    def parameters(self):
        """
        Parameters of the components

        :return: dictionary with parameters
        """
        return {p.name: p.val for p in self._parameters}

    def update(self, **kwargs):
        """
        Update the current value of the component
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Name of the component

        :return: a string representing the name of the component
        """
        return type(self)._name

    @classmethod
    def get_descriptors(cls):
        return cls._descriptors


@weight("W_anx")
@parameter("p1", doc="absolute weight of the corner tile")
@parameter("p2", minv=0., maxv=1., doc="relative weight of the wall tile")
@parameter("p3", minv=0., maxv=1., doc="relative weight of the central area tile")
class AnxietyComponent(Component):
    """
    Component that evaluates the current position in the maze

    This component has a weight for every type of tile in the maze.
    """

    _name = "anx"

    def __init__(self, **params):
        super().__init__(**params)

        self._maze_weights = self._create_weights(c=self.p1, w=self.p2 * self.p1, m=self.p3 * self.p2 * self.p1)

    @staticmethod
    def _create_weights(c, w, m, v=-1.0):
        return {StandardTiles.TileType.WALL: w,
                StandardTiles.TileType.CORNER: c,
                StandardTiles.TileType.CENTRE: m,
                StandardTiles.TileType.VOID: v}

    def update(self, next_tiles, **kwargs):
        self._val = np.array([self._maze_weights[ns] for ns in next_tiles])
        if self.debug:
            print(f'PositionComponent: {self._val}')


@weight("W_bcost")
@parameter("k", doc="")
@parameter("gamma", doc="")
class BiomechanicalCostComponent(Component):
    """
    Component that evaluates the direction of movement
    """

    _name = "bcost"

    def __init__(self, discrete_actions, ** params):
        super().__init__(**params)

        # cache the val as it is constant
        discrete_actions = np.array(discrete_actions)

        rv = vonmises(self.k)

        # base values
        angles = np.arctan2(discrete_actions[:, 1], discrete_actions[:, 0])
        self._cachedval = rv.pdf(angles)
        self._cachedval[np.all(discrete_actions == (0, 0), axis=1)] = 0.0  # (0, 0) is a special action that has value 0

        # gamma
        distances = np.linalg.norm(discrete_actions, ord=np.inf, axis=1)
        gammas = np.ones(len(discrete_actions)) * self.gamma
        gammas[distances < 2] = 1
        self._cachedval = self._cachedval * gammas

    def update(self, moving, is_visitable, **kwargs):

        if moving:
            self._val = self._cachedval[is_visitable]
        else:
            self._val = np.zeros(np.sum(is_visitable))

        if self.debug:
            print(f'DirectionComponent: {self._val}')


@weight("W_exp")
@parameter("xi", minv=0.006, maxv=0.6, default=0.006, doc="")
@parameter("kappa_home", minv=0.01, maxv=5.0, default=0.01, doc="")
@parameter("kappa_explo", minv=0.01, maxv=5.0, default=0.01, doc="")
@parameter("theta_explo", minv=80., maxv=120., default=80., doc="")
@parameter("theta_home", minv=30., maxv=70., default=30.0, doc="")
class ExperienceComponent(Component):
    """
    Component that evaluates the experience of the agent
    """

    _name = "exp"

    class State(IntEnum):
        EXPLO = 0
        HOME = 1

    def __init__(self, maze_shape, maze_map_exp=None, **params):
        super().__init__(**params)

        # mental state
        self._ment_state = self.State.EXPLO

        if maze_map_exp is not None:
            self._maze_map_exp = np.array(maze_map_exp)
        else:
            self._maze_map_exp = np.zeros(maze_shape)

        self._familiarity = 0.0

    def update(self, x, y, next_positions, **kwargs):

        # update on the number of visits in the current position of the agent in the maze
        if self.debug:
            print(f'x {x}, y {y}')
            print(f'self._maze_map_exp {self._maze_map_exp}')
            print(f'type self._maze_map_exp {type(self._maze_map_exp)}')
        self._maze_map_exp[x, y] += 1

        # update on the familiarity measure based on the current position of the agent in the maze
        num_visit_current_state = self._maze_map_exp[x, y]
        self._familiarity = (1 - self.xi) * self._familiarity + self.xi * num_visit_current_state

        # update of the mental state
        if self._ment_state == self.State.HOME and self._familiarity >= self.theta_explo:
            self._ment_state = self.State.EXPLO
        elif self._ment_state == self.State.EXPLO and self._familiarity < self.theta_home:
            self._ment_state = self.State.HOME

        if self.debug:
            print(f"new mental state: {self._ment_state}")
            print("self.familiarity: {}".format(self._familiarity))

        # computation of the new value
        num_next_positions = len(next_positions)
        self._val = np.zeros(num_next_positions)
        for pid, pos in enumerate(next_positions):
            if self._ment_state == self.State.EXPLO:
                self._val[pid] = np.exp(-self.kappa_explo * self._maze_map_exp[pos[0], pos[1]])
            else:
                self._val[pid] = 1.0 - np.exp(-self.kappa_home * self._maze_map_exp[pos[0], pos[1]])

        if self.debug:
            print(f'ExperienceComponent: {self._val}')


@weight("W_bper")
@parameter("bp", doc="")
@parameter("Wm", doc="")
@parameter("Wnm", doc="")
class BiomechPersistenceComponent(Component):
    """
    Component that evaluates the tendence to continue the same motion (moving or stopping)
    """

    _name = "bper"

    def __init__(self, discrete_actions, **params):
        super().__init__(**params)

        # cache the value
        discrete_actions = np.array(discrete_actions)

        self._cachedval_mov = np.array([self.bp] * len(discrete_actions))
        self._cachedval_mov[np.all(discrete_actions == (0, 0), axis=1)] = self.Wm * self.bp

        self._cachedval_not_mov = np.array([self.Wnm * self.bp] * len(discrete_actions))
        self._cachedval_not_mov[np.all(discrete_actions == (0, 0), axis=1)] = self.bp

    def update(self, moving, is_visitable, **kwargs):

        if moving:
            self._val = self._cachedval_mov[is_visitable]
        else:
            self._val = self._cachedval_not_mov[is_visitable]

        if self.debug:
            print(f'BiomechPersistenceComponent: {self._val}')


@weight("W_values")
class ValueTableComponent(Component):
    """
    Component that uses a previously computed value matrix to compute the values for the next states
    """

    _name = "v_table"

    def __init__(self, v_table, **params):
        super().__init__(**params)
        # normalize between 0 and 1 the values
        max_v = np.max(v_table)
        min_v = np.min(v_table)
        if max_v == min_v == 0:
            self._v_table = np.zeros(v_table.shape)
        else:
            self._v_table = (v_table - min_v) / (max_v - min_v) - 1

    def update(self, next_positions, **kwargs):

        self._val = np.zeros(len(next_positions))
        for pid, pos in enumerate(next_positions):
            self._val[pid] = self._v_table[pos[0], pos[1]]

        if self.debug:
            print(f'ValuesComponent: {self._val}')
