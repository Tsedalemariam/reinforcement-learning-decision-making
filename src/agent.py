import numpy as np


class QLearningAgent:
    """
    A simple reinforcement learning agent that learns action values
    through trial and error, using an epsilon-greedy strategy.
    """

    def __init__(self, n_actions, epsilon=0.1, learning_rate=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon        # probability of exploring
        self.lr = learning_rate       # how fast beliefs update
        # current belief about each action's value
        self.q_values = np.zeros(n_actions)

    def choose_action(self):
        """
        Epsilon-greedy action selection:
        - with probability epsilon: explore (random action)
        - otherwise: exploit (choose the currently best-known action)
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        """
        Rescorla-Wagner / Q-learning update rule:
        new_estimate = old_estimate + learning_rate * (reward - old_estimate)

        The term (reward - old_estimate) is the "prediction error" —
        the gap between what we expected and what we got.
        """
        prediction_error = reward - self.q_values[action]
        self.q_values[action] += self.lr * prediction_error
