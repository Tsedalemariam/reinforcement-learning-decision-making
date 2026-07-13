import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/simulation_results.csv")

# 1. Learning curve: rolling average reward over time
plt.figure(figsize=(8, 4))
plt.plot(df["trial"], df["reward"].rolling(20).mean())
plt.xlabel("Trial")
plt.ylabel("Average Reward (rolling window)")
plt.title("Learning Curve: Reward Over Time")
plt.savefig("../results/learning_curve.png")
plt.show()

# 2. Q-value convergence: does the agent learn the true reward ranking?
plt.figure(figsize=(8, 4))
plt.plot(df["trial"], df["q0"], label="Q(Machine 0)")
plt.plot(df["trial"], df["q1"], label="Q(Machine 1)")
plt.plot(df["trial"], df["q2"], label="Q(Machine 2)")
plt.xlabel("Trial")
plt.ylabel("Estimated Value")
plt.title("Q-Value Convergence")
plt.legend()
plt.savefig("../results/q_convergence.png")
plt.show()

# 3. Action selection frequency: does it converge on the best machine?
plt.figure(figsize=(6, 4))
df["action"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Action (Machine)")
plt.ylabel("Times Chosen")
plt.title("Action Selection Frequency")
plt.savefig("../results/action_frequency.png")
plt.show()
