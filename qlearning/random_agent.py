import numpy as np


class RandomAgent:

    """
    An agent that selects among the available actions uniformly at random
    """

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def observe(self, s0, a, r, s1, done):
        pass  # do nothing

    def get_action(self, state):
        return np.random.randint(self.n_actions)
