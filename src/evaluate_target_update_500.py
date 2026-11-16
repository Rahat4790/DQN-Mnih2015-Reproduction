
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

from pathlib import Path

from src.environment import make_env, get_env_info
from src.network import QNetwork


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_one_model(training_seed, num_episodes=100):

    evaluation_seed = 1000

    set_seed(evaluation_seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = make_env()

    state_dim, action_dim = get_env_info(env)

    q_network = QNetwork(
        state_dim,
        action_dim
    ).to(device)

    # Load the correct target update interval = 500 model
    model_path = Path(
        f"results/sensitivity/target_update/interval_500/seed_{training_seed}/online_network.pth"
    )

    if not model_path.exists():
        env.close()
        raise FileNotFoundError(model_path)

    q_network.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    q_network.eval()

    episode_rewards = []

    for episode in range(num_episodes):

        state, info = env.reset(
            seed=evaluation_seed + episode
        )

        total_reward = 0
        done = False

        while not done:

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():

                q_values = q_network(state_tensor)
                action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

    env.close()

    return episode_rewards


def evaluate_all_models():

    training_seeds = [42, 123, 2025, 7, 21]
    num_episodes = 100

    # Correct output directory
    output_dir = Path(
        "results/sensitivity/target_update/interval_500/evaluation"
    )

    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []

    print("\nTARGET UPDATE INTERVAL SENSITIVITY EVALUATION")
    print("=============================================")

    for training_seed in training_seeds:

        print(f"\nEvaluating training seed: {training_seed}")

        rewards = evaluate_one_model(
            training_seed,
            num_episodes
        )

        all_results[training_seed] = rewards

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)

        summary_rows.append({
            "training_seed": training_seed,
            "evaluation_episodes": num_episodes,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward
        })

        np.save(
            output_dir / f"seed_{training_seed}_rewards.npy",
            np.array(rewards)
        )

        print(f"Mean reward: {mean_reward:.2f}")
        print(f"Standard deviation: {std_reward:.2f}")
        print(f"Minimum reward: {min_reward:.0f}")
        print(f"Maximum reward: {max_reward:.0f}")

    # Save summary CSV
    with open(
        output_dir / "evaluation_summary.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "training_seed",
                "evaluation_episodes",
                "mean_reward",
                "std_reward",
                "min_reward",
                "max_reward"
            ]
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # Plot evaluation rewards
    plt.figure(figsize=(12, 6))

    for seed, rewards in all_results.items():

        plt.plot(
            range(1, num_episodes + 1),
            rewards,
            marker=".",
            markersize=3,
            linewidth=1,
            label=f"Seed {seed}"
        )

    plt.xlabel("Evaluation Episode")
    plt.ylabel("Total Reward")
    plt.title("Greedy Evaluation: Target Update Interval = 500")

    plt.grid(True)
    plt.legend()

    plt.savefig(
        figure_dir / "target_update_500_evaluation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Final summary
    seed_means = [
        row["mean_reward"]
        for row in summary_rows
    ]

    print("\n========== FINAL SUMMARY ==========")

    for row in summary_rows:

        print(
            f"Seed {row['training_seed']:4d}: "
            f"Mean = {row['mean_reward']:.2f}, "
            f"Std = {row['std_reward']:.2f}"
        )

    print("-----------------------------------")
    print(f"Mean across seeds: {np.mean(seed_means):.2f}")

    print(
        "Standard deviation across seeds:",
        f"{np.std(seed_means, ddof=1):.2f}"
    )

    print(f"Median: {np.median(seed_means):.2f}")

    print("===================================")
    print("\nEvaluation completed successfully!")

    print(f"\nResults saved in: {output_dir}")
    print(f"Plot saved in: {figure_dir}")


if __name__ == "__main__":

    evaluate_all_models()
