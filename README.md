# Learning to Make Better Decisions: A Reinforcement Learning Simulation of Human Reward-Based Learning

## Overview

This project implements a simple **reinforcement learning (RL)** simulation — a type of machine learning where an agent learns by trial and error, receiving rewards or feedback for its actions rather than being told the "correct answer" directly (unlike most standard machine learning, which learns from labeled examples).

The simulation models a classic problem called a **multi-armed bandit task**: imagine a row of slot machines, each with a different, unknown average payout. An agent repeatedly chooses among them, receives a reward each time, and must figure out — purely from experience — which machine is best. This is one of the simplest possible settings in reinforcement learning, and it captures the core challenge of learning under uncertainty.

## Motivation

This project explores the computational principles behind reward-based learning and decision-making: how an agent builds and updates beliefs about the value of its choices, and how it balances **exploration** (trying new, uncertain options to gather information) against **exploitation** (choosing the option that currently seems best).

The learning rule used here is directly related to the **Rescorla-Wagner model**, a foundational theory of associative learning in psychology. It updates beliefs based on a **prediction error** — the gap between the reward an agent expected and the reward it actually received. This same mathematical idea underlies both classical learning theory and modern reinforcement learning algorithms, which is what makes this simple simulation a meaningful bridge between the two fields.

This project was built ahead of joining the **Learning and Decision Neuroscience Lab** at UC Irvine, directed by Dr. Mimi Liljeholm, whose research examines how humans discover and represent the predictive structure of their environment and how that shapes decision-making.

## Project Structure
reinforcement-learning-decision-making/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── environment.py        # Defines the bandit task (the "world" the agent acts in)
│   ├── agent.py               # The learning agent (epsilon-greedy Q-learning)
│   ├── simulation.py          # Runs one agent through many trials, logs results
│   ├── plot_results.py        # Generates learning curve and Q-value plots
│   ├── compare_agents.py      # Runs the simulation across multiple exploration rates
│   └── plot_comparison.py     # Plots learning curves across exploration rates
│
└── results/
├── simulation_results.csv
├── agent_comparison.csv
├── learning_curve.png
├── q_convergence.png
├── action_frequency.png
└── epsilon_comparison.png

## How It Works

**Environment (`environment.py`)**
Defines the task itself, separate from whoever is making decisions in it. It stores the *true* average reward of each arm (hidden from the agent) and, each time the agent picks an arm, returns a reward sampled from a normal (bell-curve) distribution centered on that arm's true value. This models the fact that real-world outcomes are noisy — even a good choice doesn't give exactly the same reward every time.

**Agent (`agent.py`)**
The learner. It keeps a running estimate of each arm's value (called a **Q-value**, starting at zero for all arms — the agent begins knowing nothing). It chooses actions using an **epsilon-greedy policy**:
- Most of the time, it picks whichever arm currently looks best (**exploitation**).
- With a small probability, called **epsilon**, it picks a random arm instead (**exploration**), so it doesn't miss out on discovering a better option.

After each choice, it updates its belief using this rule:
Q(a) ← Q(a) + α × (reward − Q(a))

Here, α (alpha) is the **learning rate** — how much each new experience shifts the belief — and `(reward − Q(a))` is the **prediction error**: how far off the agent's expectation was. This single equation is the entire "learning" in this project.

**Simulation (`simulation.py`)**
Runs the agent through many trials (1000 by default), recording the action taken, reward received, and current value estimates at every step into a table using **pandas** (a Python library for working with tabular data, similar to a spreadsheet). This is saved as a CSV file — the same format commonly used for real behavioral experiment data.

**Visualization (`plot_results.py`)**
Uses **Matplotlib** (a Python plotting library) to turn that table into three plots:
- **Learning curve** — average reward over time, smoothed with a rolling average
- **Q-value convergence** — how the agent's belief about each arm evolves and settles
- **Action selection frequency** — how often each arm was ultimately chosen

**Exploration Rate Comparison (`compare_agents.py`, `plot_comparison.py`)**
Repeats the entire simulation three times with different epsilon values (0.01, 0.1, 0.3 — i.e., low, moderate, and high exploration) to directly compare how the exploration-exploitation tradeoff affects both how *fast* the agent learns and how *well* it ultimately performs.

## Getting Started

**Requirements:** Python 3.10+

```bash
git clone <your-repo-url>
cd reinforcement-learning-decision-making
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run the core simulation:**
```bash
cd src
python simulation.py
python plot_results.py
```

**Run the exploration-rate comparison:**
```bash
python compare_agents.py
python plot_comparison.py
```

All outputs (CSVs and plots) are saved to the `results/` folder.

## Results

With a 3-armed bandit (true reward values: 5, 8, and 3), the agent reliably learns to favor the highest-value arm.

![Learning Curve](results/learning_curve.png)

![Q-Value Convergence](results/q_convergence.png)

**Effect of exploration rate:**

![Epsilon Comparison](results/epsilon_comparison.png)

With **epsilon = 0.01** (rarely explores), the agent explored too infrequently to reliably discover the best-performing arm, and its average reward stayed flat near 5 for the entire 1000 trials — a clear demonstration of the risk of too little exploration: the agent settles for a mediocre option simply because it never tries the better one enough times to notice it's better.

With **epsilon = 0.1** (moderate exploration), performance rose sharply around trial 250–300 as the agent discovered the optimal arm, then stabilized at the highest average reward of the three conditions (~8).

With **epsilon = 0.3** (frequent exploration), the agent found the optimal arm considerably faster (by roughly trial 100–150), but its long-run average reward settled lower (~7), because it kept sampling worse arms 30% of the time even after it had effectively already learned which arm was best.

**Takeaway:** these results illustrate the exploration-exploitation tradeoff at the heart of reinforcement learning — too little exploration risks never discovering the best option; too much exploration sacrifices long-run performance even after the best option is known. A moderate exploration rate achieved the best overall balance in this setting.

## What This Project Demonstrates

- Implementation of a foundational reinforcement learning algorithm (Q-learning with epsilon-greedy exploration) from first principles, using only NumPy
- Structuring simulation output as tabular data with pandas, in a format analogous to real trial-by-trial behavioral data
- Visualizing learning dynamics with Matplotlib
- An empirical investigation of the exploration-exploitation tradeoff, a central theme in both reinforcement learning and human/animal decision-making research

## Possible Extensions

- Compare the Q-learning agent against a simpler heuristic strategy (e.g. "win-stay, lose-shift": repeat the last action if it paid off, switch otherwise) to explore how more "habitual" strategies differ from value-based learning
- Introduce non-stationary reward probabilities (arms whose true value changes over time) to study how the agent adapts to a changing environment
- Fit the model to real behavioral choice data and use scikit-learn to compare its fit against alternative decision-making models

## Author

Tsedalemariam Getu
Addis Ababa University
