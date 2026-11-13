
import os
import pandas as pd
import matplotlib.pyplot as plt


# Training seeds
seeds = [42, 123, 2025, 7, 21]

# Create figure
plt.figure(figsize=(12, 6))


# Load and plot loss data for each seed
for seed in seeds:

    file_path = os.path.join(
        "results",
        "sensitivity",
        "raw",
        f"baseline_seed_{seed}",
        "episode_metrics.csv"
    )

    df = pd.read_csv(file_path)

    # Calculate 50-episode moving average loss
    df["smoothed_loss"] = (
        df["average_loss"].rolling(
            window=50,
            min_periods=1
        ).mean()
    )

    plt.plot(
        df["episode"],
        df["smoothed_loss"],
        label=f"Seed {seed}"
    )


# Graph formatting
plt.xlabel("Episode")
plt.ylabel("50-Episode Moving Average Loss")

plt.title(
    "Vanilla DQN: Training Loss Across Five Random Seeds"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# Save figure
output_dir = os.path.join(
    "results",
    "sensitivity",
    "figures"
)

os.makedirs(output_dir, exist_ok=True)

plt.savefig(
    os.path.join(output_dir, "seed_loss_comparison.png"),
    dpi=300
)

# Display graph
plt.show()

print("Loss comparison graph saved successfully!")
