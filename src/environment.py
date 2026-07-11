import numpy as np


class BanditEnvironment:

    def __init__(self):
        # True reward values of each option
        # The agent does not know these
        self.rewards = {
            0: 5,
            1: 8
        }

    def step(self, action):
        """
        Agent chooses an action.
        Environment returns a reward.
        """

        true_reward = self.rewards[action]

        # Add randomness (uncertainty)
        reward = np.random.normal(
            true_reward,
            1
        )

        return reward
