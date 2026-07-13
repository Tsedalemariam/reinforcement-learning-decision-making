import numpy as np
import pandas as pd
from environment import BanditEnvironment
from agent import QLearningAgent


def run_simulation(true_rewards=[5, 8, 3], epsilon=0.1, learning_rate=0.1, n_trials=1000):
    env = BanditEnvironment(true_rewards)
    agent = QLearningAgent(n_actions=env.n_actions(),
                           epsilon=epsilon, learning_rate=learning_rate)

    records = []  # will hold one row per trial

    for trial in range(n_trials):
        action = agent.choose_action()
        reward = env.step(action)
        agent.update(action, reward)

        records.append({
            "trial": trial,
            "action": action,
            "reward": reward,
            "q0": agent.q_values[0],
            "q1": agent.q_values[1],
            "q2": agent.q_values[2],
        })

    df = pd.DataFrame(records)
    return df, agent


if __name__ == "__main__":
    df, agent = run_simulation()
    df.to_csv("../results/simulation_results.csv", index=False)
    print("Final learned Q-values:", agent.q_values)
    print("True rewards were:", [5, 8, 3])
    print("Saved results to results/simulation_results.csv")
