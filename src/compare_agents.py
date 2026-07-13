import numpy as np
import pandas as pd
from environment import BanditEnvironment
from agent import QLearningAgent


def run_simulation(true_rewards, epsilon, learning_rate, n_trials):
    """
    Runs one full simulation for a given epsilon value
    and returns a dataframe of results, tagged with that epsilon.
    """
    env = BanditEnvironment(true_rewards)
    agent = QLearningAgent(n_actions=env.n_actions(),
                           epsilon=epsilon, learning_rate=learning_rate)

    records = []
    for trial in range(n_trials):
        action = agent.choose_action()
        reward = env.step(action)
        agent.update(action, reward)

        records.append({
            "epsilon": epsilon,
            "trial": trial,
            "action": action,
            "reward": reward,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    true_rewards = [5, 8, 3]
    n_trials = 1000
    epsilons_to_test = [0.01, 0.1, 0.3]

    all_results = []

    for eps in epsilons_to_test:
        df = run_simulation(true_rewards, epsilon=eps,
                            learning_rate=0.1, n_trials=n_trials)
        all_results.append(df)
        print(f"Finished epsilon={eps}")

    # Combine all three runs into one dataframe, one column tells them apart
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv("../results/agent_comparison.csv", index=False)
    print("Saved comparison results to results/agent_comparison.csv")
