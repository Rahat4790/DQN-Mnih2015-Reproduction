
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

from pathlib import Path

from src.environment import make_env, get_env_info
from src.network import QNetwork


# ============================================================
# 1. RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. EVALUATE ONE TRAINED MODEL
# ============================================================

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

    model_path = Path(
        f"results/sensitivity/diagnostics/seed_{training_seed}/online_network.pth"
    )

    if not model_path.exists():
        env.close()
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

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


# ============================================================
# 3. EVALUATE ALL FIVE TRAINING SEEDS
# ============================================================

def evaluate_all_models():

    training_seeds = [42, 123, 2025, 7, 21]

    num_episodes = 100

    output_dir = Path(
        "results/sensitivity/diagnostics/evaluation"
    )

    figure_dir = output_dir / "figures"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = {}

    summary_rows = []

    print("\nGREEDY EVALUATION OF DIAGNOSTIC MODELS")
    print("======================================")

    print("Environment: CartPole-v1")
    print("Evaluation episodes per model:", num_episodes)
    print("Exploration rate: 0")
    print("Evaluation reset seeds: 1000–1099")

    for training_seed in training_seeds:

        print(
            f"\nEvaluating training seed: {training_seed}"
        )

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
            output_dir / f"seed_{training_seed}_evaluation_rewards.npy",
            np.array(rewards)
        )

        print(f"Mean reward: {mean_reward:.2f}")
        print(f"Standard deviation: {std_reward:.2f}")
        print(f"Minimum reward: {min_reward:.0f}")
        print(f"Maximum reward: {max_reward:.0f}")

    # --------------------------------------------------------
    # SAVE SUMMARY CSV
    # --------------------------------------------------------

    summary_path = output_dir / "evaluation_summary.csv"

    with open(
        summary_path,
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

    # --------------------------------------------------------
    # SAVE COMBINED REWARD FIGURE
    # --------------------------------------------------------

    plt.figure(figsize=(12, 6))

    for training_seed, rewards in all_results.items():

        plt.plot(
            range(1, num_episodes + 1),
            rewards,
            marker=".",
            markersize=3,
            linewidth=1,
            label=f"Training seed {training_seed}"
        )

    plt.xlabel("Evaluation Episode")
    plt.ylabel("Total Reward")

    plt.title(
        "Greedy Evaluation of Diagnostic DQN Models"
    )

    plt.grid(True)
    plt.legend()

    plt.savefig(
        figure_dir / "diagnostic_greedy_evaluation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # PRINT FINAL SUMMARY
    # --------------------------------------------------------

    seed_means = [
        row["mean_reward"]
        for row in summary_rows
    ]

    print("\n========== FINAL EVALUATION SUMMARY ==========")

    for row in summary_rows:

        print(
            f"Seed {row['training_seed']:4d}: "
            f"Mean = {row['mean_reward']:.2f}, "
            f"Std = {row['std_reward']:.2f}"
        )

    print("----------------------------------------------")

    print(
        f"Mean across training seeds: "
        f"{np.mean(seed_means):.2f}"
    )

    print(
        f"Standard deviation across training seeds: "
        f"{np.std(seed_means, ddof=1):.2f}"
    )

    print(
        f"Median across training seeds: "
        f"{np.median(seed_means):.2f}"
    )

    print("==============================================")

    print("\nEvaluation completed successfully!")


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":

    evaluate_all_models()
