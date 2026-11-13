
import os
import pandas as pd
import numpy as np

seeds = [42, 123, 2025, 7, 21]

all_results = []

for seed in seeds:

    file_path = os.path.join(
        "results",
        "sensitivity",
        "raw",
        f"baseline_seed_{seed}",
        "episode_metrics.csv"
    )

    df = pd.read_csv(file_path)

    print(f"\nSeed: {seed}")
    print("Completed episodes:", len(df))
    print("Mean reward:", round(df["reward"].mean(), 2))
    print("Maximum reward:", df["reward"].max())
    print("Final 10 episode mean:",
          round(df["reward"].tail(10).mean(), 2))

    all_results.append({
        "seed": seed,
        "mean_reward": df["reward"].mean(),
        "max_reward": df["reward"].max(),
        "final_10_mean": df["reward"].tail(10).mean()
    })

results_df = pd.DataFrame(all_results)

print("\n========== OVERALL BASELINE ==========")

for metric in ["mean_reward", "max_reward", "final_10_mean"]:

    print(f"\n{metric}:")
    print("Mean:", round(results_df[metric].mean(), 2))
    print("Standard deviation:",
          round(results_df[metric].std(ddof=1), 2))
