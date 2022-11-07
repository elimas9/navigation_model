import numpy as np


class ExplorationModel:

    @property
    def parameters(self):
        """
        Get the parameters of the model, including parameters of the components

        :return: dictionary with all parameters
        """
        raise NotImplementedError()

    def decision_making(self, **kwargs):
        """
        Choose the next position

        :return: next position chosen
        """
        raise NotImplementedError()


class BehaviouralModel(ExplorationModel):
    """
    An exploration model that bases its decisions on behavioural components.

    It is defined by some components, an initial discrete position, and a trade-off parameter between
    exploration and exploitation.
    The set of values for the possible next states is computed from the values given by the components.
    """

    def __init__(self, components, init_pos_discrete, beta=None, seed=None, debug=False):
        """
        Create a behavioural model

        :param components: list of behavioural components of the model
        :param init_pos_discrete: initial discrete position of the agent
        :param beta: float describing the parameter of the model describing the trade-off between exploration and
        exploitation
        :param seed: seed for random choices
        :param debug: enable debug prints
        """
        self._components = components

        # used to compute moving True/False
        self._prev_pos = init_pos_discrete
        self._moving = False

        # decision-making
        self._beta = beta
        self._rng = np.random.default_rng(seed)

        # parameters dictionary
        self._model_parameters = {
            "beta": beta
        }
        for comp_pars in [c.parameters for c in self._components]:
            for k, v in comp_pars.items():
                self._model_parameters[k] = v

        self._debug = debug

    @property
    def parameters(self):
        return self._model_parameters

    def decision_making(self, next_positions_cont, **kwargs):
        """
        Choose the next position

        This should take only the possible next positions as arguments, as it will not perform any check.

        :param next_positions_cont: next positions (continuous)
        :param kwargs: other arguments for behavioural components
        :return: new position chosen
        """
        # update values for components
        for comp in self._components:
            comp.update(moving=self._moving,
                        **kwargs)

        # choose next position based on values
        val_fin = np.zeros(len(next_positions_cont))
        for c in self._components:
            val_fin = val_fin + c.weighted_values
        val_fin = self._beta * val_fin
        if self._debug:
            print(f'val_fin: {val_fin}')

        # softmax
        val_fin = val_fin - np.max(val_fin)  # this was useful but I don't remember why
        p_softmax = np.exp(val_fin) / np.sum(np.exp(val_fin))
        if self._debug:
            print(f'P_softMax: {p_softmax}')

        # choice
        new_pos = self._rng.choice(next_positions_cont, p=p_softmax)

        # moving/not moving
        self._moving = np.all(self._prev_pos == new_pos)
        if self._debug:
            if self._moving:
                print(f"*** moving from: {self._prev_pos} to {new_pos} ***")
            else:
                print(f"*** same place: {self._prev_pos} = {new_pos} ***")
        self._prev_pos = new_pos

        return new_pos


class RandomModel(ExplorationModel):
    """
    An exploration model that chooses the next position using a random process
    """

    def __init__(self, seed=None, debug=False):
        """
        Create a random decision maker

        :param seed: seed for random choices
        :param debug: enable debug prints
        """
        self._rng = np.random.default_rng(seed)
        self._debug = debug

    @property
    def parameters(self):
        return {}

    def decision_making(self, next_positions_cont, **kwargs):
        """
        Choose the next position

        This should take only the possible next positions as arguments, as it will not perform any check.

        :param next_positions_cont: next positions (continuous)
        :param kwargs: unused
        :return: new position chosen
        """
        next_pos = self._rng.choice(next_positions_cont)
        if self._debug:
            print("move: {}".format(next_pos))
        return next_pos
