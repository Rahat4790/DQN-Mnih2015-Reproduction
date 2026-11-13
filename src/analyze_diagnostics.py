
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEEDS = [42, 123, 2025, 7, 21]

BASE_DIR = os.path.join(
    "results",
    "sensitivity",
    "diagnostics"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "analysis"
)

SMOOTHING_WINDOW = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD EXPERIMENTAL DATA
# ============================================================

def load_experiment_data(seed):

    seed_dir = os.path.join(
        BASE_DIR,
        f"seed_{seed}"
    )

    episode_file = os.path.join(
        seed_dir,
        "episode_metrics.csv"
    )

    diagnostic_file = os.path.join(
        seed_dir,
        "diagnostic_metrics.csv"
    )

    if not os.path.exists(episode_file):
        raise FileNotFoundError(episode_file)

    if not os.path.exists(diagnostic_file):
        raise FileNotFoundError(diagnostic_file)

    episode_df = pd.read_csv(episode_file)

    diagnostic_df = pd.read_csv(diagnostic_file)

    # Merge using episode number
    df = pd.merge(
        episode_df,
        diagnostic_df,
        on="episode",
        how="inner",
        validate="one_to_one"
    )

    if len(df) != len(episode_df):
        raise ValueError(
            f"Episode mismatch for seed {seed}"
        )

    df = df.sort_values("episode").reset_index(drop=True)

    print(
        f"Seed {seed}: "
        f"{len(df)} episodes loaded"
    )

    return df


# ============================================================
# 3. LOAD ALL FIVE SEEDS
# ============================================================

def load_all_experiments():

    experiments = {}

    for seed in SEEDS:

        experiments[seed] = load_experiment_data(seed)

    return experiments


# ============================================================
# 4. MOVING AVERAGE
# ============================================================

def smooth_data(series):

    return series.rolling(
        window=SMOOTHING_WINDOW,
        min_periods=1
    ).mean()


# ============================================================
# 5. GENERIC PLOT FUNCTION
# ============================================================

def plot_metric(
    experiments,
    columns,
    title,
    ylabel,
    filename
):

    plt.figure(figsize=(11, 6))

    for seed, df in experiments.items():

        for column, label in columns:

            values = smooth_data(df[column])

            if len(columns) == 1:

                plt.plot(
                    df["episode"],
                    values,
                    label=f"Seed {seed}"
                )

            else:

                # Plot each metric with a seed-specific label
                plt.plot(
                    df["episode"],
                    values,
                    label=f"Seed {seed} - {label}"
                )

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )

    plt.close()

    print(f"Saved plot: {filename}")


# ============================================================
# 6. PLOT REWARD
# ============================================================

def plot_rewards(experiments):

    plot_metric(
        experiments,
        [("reward", "Reward")],
        "Training Reward Across Random Seeds",
        "Episode Reward",
        "01_training_rewards.png"
    )


# ============================================================
# 7. PLOT TRAINING LOSS
# ============================================================

def plot_losses(experiments):

    plot_metric(
        experiments,
        [("average_loss", "Loss")],
        "Training Loss Across Random Seeds",
        "Average Training Loss",
        "02_training_losses.png"
    )


# ============================================================
# 8. PLOT CURRENT AND TARGET Q-VALUES
# ============================================================

def plot_q_values(experiments):

    plt.figure(figsize=(11, 6))

    for seed, df in experiments.items():

        current_q = smooth_data(
            df["mean_current_q"]
        )

        target_q = smooth_data(
            df["mean_target_q"]
        )

        plt.plot(
            df["episode"],
            current_q,
            label=f"Seed {seed} - Current Q"
        )

        plt.plot(
            df["episode"],
            target_q,
            linestyle="--",
            label=f"Seed {seed} - Target Q"
        )

    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.title("Current Q-values and Target Q-values")

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    plt.tight_layout()

    filename = "03_q_value_comparison.png"

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )

    plt.close()

    print(f"Saved plot: {filename}")


# ============================================================
# 9. PLOT MAXIMUM ABSOLUTE CURRENT Q-VALUES
# ============================================================

def plot_max_current_q(experiments):

    plot_metric(
        experiments,
        [("max_abs_current_q", "Maximum Absolute Current Q")],
        "Maximum Absolute Selected-Action Q-value",
        "Maximum Absolute Q-value",
        "04_max_abs_current_q.png"
    )


# ============================================================
# 10. PLOT TD ERRORS
# ============================================================

def plot_td_errors(experiments):

    plt.figure(figsize=(11, 6))

    for seed, df in experiments.items():

        mean_error = smooth_data(
            df["mean_abs_td_error"]
        )

        max_error = smooth_data(
            df["max_abs_td_error"]
        )

        plt.plot(
            df["episode"],
            mean_error,
            label=f"Seed {seed} - Mean TD Error"
        )

        plt.plot(
            df["episode"],
            max_error,
            linestyle="--",
            label=f"Seed {seed} - Maximum TD Error"
        )

    plt.xlabel("Episode")
    plt.ylabel("Absolute TD Error")

    plt.title("Temporal Difference Error Across Seeds")

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    plt.tight_layout()

    filename = "05_td_error_comparison.png"

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )

    plt.close()

    print(f"Saved plot: {filename}")


# ============================================================
# 11. PLOT GRADIENT NORMS
# ============================================================

def plot_gradient_norms(experiments):

    plot_metric(
        experiments,
        [("gradient_norm", "Gradient Norm")],
        "Gradient Norm Across Random Seeds",
        "Gradient Norm Before Clipping",
        "06_gradient_norms.png"
    )


# ============================================================
# 12. GENERATE STATISTICAL SUMMARY
# ============================================================

def generate_summary(experiments):

    summary_rows = []

    metrics = [
        "reward",
        "average_loss",
        "mean_current_q",
        "max_abs_current_q",
        "mean_target_q",
        "mean_abs_td_error",
        "max_abs_td_error",
        "gradient_norm"
    ]

    for seed, df in experiments.items():

        row = {
            "seed": seed,
            "episodes": len(df)
        }

        for metric in metrics:

            values = df[metric].dropna()

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_max"] = values.max()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    output_file = os.path.join(
        OUTPUT_DIR,
        "diagnostic_summary.csv"
    )

    summary_df.to_csv(
        output_file,
        index=False
    )

    print("\nSTATISTICAL SUMMARY")
    print("=" * 80)

    display_columns = [
        "seed",
        "episodes",
        "reward_mean",
        "reward_std",
        "average_loss_mean",
        "mean_abs_td_error_mean",
        "gradient_norm_mean"
    ]

    print(
        summary_df[display_columns].round(4).to_string(
            index=False
        )
    )

    print(f"\nSummary saved: {output_file}")


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("\nLoading diagnostic experiments...\n")

    experiments = load_all_experiments()

    print("\nGenerating diagnostic plots...\n")

    plot_rewards(experiments)

    plot_losses(experiments)

    plot_q_values(experiments)

    plot_max_current_q(experiments)

    plot_td_errors(experiments)

    plot_gradient_norms(experiments)

    print("\nGenerating statistical summary...\n")

    generate_summary(experiments)

    print("\nDiagnostic analysis completed successfully!")

    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":

    main()
