class Session:

    def __init__(self, trajectory, sampling_time, reward=None, new_sampling_time=None):
        pass

    @property
    def trajectory(self):
        pass

    @property
    def reward(self):
        pass

    @property
    def reward_it(self):
        pass

    @property
    def reward_times(self):
        pass

    def to_dict(self):
        # sampled trajectory
        # new sampling time
        # reward_it
        pass

    @classmethod
    def from_dict(cls, session_dict):
        pass

    def plot(self, ax, c="b", c_reward=""):
        pass

    def _expand_reward(self):
        # set full reward vector from reward iterations
        pass

def correcting_measurements_bias(cont_pos, maze):
    pass
