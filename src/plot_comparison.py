import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/agent_comparison.csv")

plt.figure(figsize=(9, 5))

for eps in sorted(df["epsilon"].unique()):
    subset = df[df["epsilon"] == eps]
    rolling_avg = subset["reward"].rolling(20).mean()
    plt.plot(subset["trial"], rolling_avg, label=f"epsilon = {eps}")

plt.xlabel("Trial")
plt.ylabel("Average Reward (rolling window)")
plt.title("Effect of Exploration Rate (epsilon) on Learning")
plt.legend()
plt.savefig("../results/epsilon_comparison.png")
plt.show()
