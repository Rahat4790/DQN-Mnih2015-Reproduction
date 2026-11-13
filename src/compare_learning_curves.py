
import os
import pandas as pd
import matplotlib.pyplot as plt

seeds = [42, 123, 2025, 7, 21]

plt.figure(figsize=(12, 6))

for seed in seeds:

    file_path = os.path.join(
        "results",
        "sensitivity",
        "raw",
        f"baseline_seed_{seed}",
        "episode_metrics.csv"
    )

    df = pd.read_csv(file_path)

    df["smoothed_reward"] = (
        df["reward"].rolling(
            window=50,
            min_periods=1
        ).mean()
    )

    plt.plot(
        df["episode"],
        df["smoothed_reward"],
        label=f"Seed {seed}"
    )

plt.xlabel("Episode")
plt.ylabel("50-Episode Moving Average Reward")
plt.title("Vanilla DQN: Training Performance Across Seeds")

plt.axhline(
    y=475,
    linestyle="--",
    label="Solved threshold (475)"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

os.makedirs("results/sensitivity/figures", exist_ok=True)

plt.savefig(
    "results/sensitivity/figures/seed_comparison.png",
    dpi=300
)

plt.show()
