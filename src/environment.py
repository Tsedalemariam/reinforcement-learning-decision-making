import numpy as np


class BanditEnvironment:
    """
    Simulates a two-armed bandit environment.
    Each action corresponds to choosing one of several slot machines.
    The agent receives a reward sampled from a normal distribution
    centered on the machine's true (hidden) average reward.
    """

    def __init__(self, true_rewards=None):
        # Hidden average reward of each machine — the agent never sees these
        self.true_rewards = true_rewards if true_rewards is not None else [
            5, 8, 3]

    def n_actions(self):
        return len(self.true_rewards)

    def step(self, action):
        """
        Returns a noisy reward for the selected action.
        """
        true_reward = self.true_rewards[action]
        reward = np.random.normal(true_reward, 1)
        return reward
