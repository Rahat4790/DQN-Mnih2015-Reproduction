
import random
import numpy as np
import torch
import csv
import json
import os

from tqdm import tqdm

from src.environment import make_env, get_env_info
from src.dqn_agent import DQNAgent


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
# 2. TARGET UPDATE INTERVAL SENSITIVITY CONFIGURATION
# ============================================================

EXPERIMENT_CONFIG = {

    "experiment": "target_update_interval_sensitivity",
    "environment": "CartPole-v1",
    "total_steps": 100000,

    "buffer_capacity": 50000,
    "batch_size": 64,

    "gamma": 0.99,

    # Keep baseline learning rate unchanged
    "learning_rate": 0.0005,

    # Changed from baseline 1000 to 500
    "target_update_interval": 500,

    "hidden_dims": (24, 24),

    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 20000,

    "gradient_clip": 10.0,

    "seeds": [42, 123, 2025, 7, 21]
}


# ============================================================
# 3. EPSILON SCHEDULE
# ============================================================

def get_epsilon(step, config):

    start = config["epsilon_start"]
    end = config["epsilon_end"]
    decay_steps = config["epsilon_decay_steps"]

    fraction = min(step / decay_steps, 1.0)

    return start + fraction * (end - start)


# ============================================================
# 4. DIAGNOSTIC METRICS
# ============================================================

DIAGNOSTIC_KEYS = [
    "mean_current_q",
    "max_abs_current_q",
    "mean_target_q",
    "mean_abs_td_error",
    "max_abs_td_error",
    "gradient_norm"
]


# ============================================================
# 5. TRAIN ONE EXPERIMENT
# ============================================================

def train_one_run(config, seed):

    set_seed(seed)

    env = make_env(seed=seed)

    state_dim, action_dim = get_env_info(env)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        learning_rate=config["learning_rate"],
        target_update_interval=config["target_update_interval"],
        hidden_dims=tuple(config["hidden_dims"]),
        gradient_clip=config["gradient_clip"]
    )

    episode_rewards = []
    episode_losses = []
    episode_diagnostics = []

    episode_losses_current = []

    episode_diagnostics_current = {
        key: [] for key in DIAGNOSTIC_KEYS
    }

    total_steps = 0

    state, info = env.reset(seed=seed)
    episode_reward = 0

    progress = tqdm(
        total=config["total_steps"],
        desc=f"Target Update Interval 500 | Seed {seed}"
    )

    while total_steps < config["total_steps"]:

        epsilon = get_epsilon(total_steps, config)

        action = agent.select_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        agent.store_experience(
            state,
            action,
            reward,
            next_state,
            terminated
        )

        loss = agent.learn()

        if loss is not None:

            episode_losses_current.append(loss)

            if agent.last_q_stats is not None:

                for key in DIAGNOSTIC_KEYS:

                    episode_diagnostics_current[key].append(
                        agent.last_q_stats[key]
                    )

        state = next_state
        episode_reward += reward

        total_steps += 1
        progress.update(1)

        if terminated or truncated:

            episode_rewards.append(episode_reward)

            average_loss = (
                float(np.mean(episode_losses_current))
                if episode_losses_current
                else 0.0
            )

            episode_losses.append(average_loss)

            diagnostic_values = {}

            for key in DIAGNOSTIC_KEYS:

                values = episode_diagnostics_current[key]

                diagnostic_values[key] = (
                    float(np.mean(values))
                    if values
                    else 0.0
                )

            episode_diagnostics.append(diagnostic_values)

            state, info = env.reset()

            episode_reward = 0
            episode_losses_current = []

            episode_diagnostics_current = {
                key: [] for key in DIAGNOSTIC_KEYS
            }

    progress.close()
    env.close()

    clipping_stats = agent.get_gradient_clipping_stats()

    return (
        episode_rewards,
        episode_losses,
        episode_diagnostics,
        clipping_stats,
        agent
    )


# ============================================================
# 6. SAVE EXPERIMENT RESULTS
# ============================================================

def save_run_results(
    config,
    seed,
    rewards,
    losses,
    diagnostics,
    clipping_stats,
    agent
):

    output_dir = os.path.join(
        "results",
        "sensitivity",
        "target_update",
        "interval_500",
        f"seed_{seed}"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(
        os.path.join(output_dir, "config.json"),
        "w"
    ) as file:

        json.dump(config, file, indent=4)

    # Save trained model
    torch.save(
        agent.online_network.state_dict(),
        os.path.join(output_dir, "online_network.pth")
    )

    # Save episode rewards and losses
    with open(
        os.path.join(output_dir, "episode_metrics.csv"),
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "reward",
            "average_loss"
        ])

        for episode, (reward, loss) in enumerate(
            zip(rewards, losses),
            start=1
        ):

            writer.writerow([
                episode,
                reward,
                loss
            ])

    # Save diagnostic metrics
    with open(
        os.path.join(output_dir, "diagnostic_metrics.csv"),
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["episode"] + DIAGNOSTIC_KEYS
        )

        for episode, diagnostic in enumerate(
            diagnostics,
            start=1
        ):

            writer.writerow(
                [episode] + [
                    diagnostic[key]
                    for key in DIAGNOSTIC_KEYS
                ]
            )

    # Save gradient clipping summary
    with open(
        os.path.join(output_dir, "gradient_clipping_summary.csv"),
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "seed",
            "gradient_clip_threshold",
            "total_updates",
            "clipped_updates",
            "clipping_percentage",
            "mean_gradient_norm",
            "max_gradient_norm"
        ])

        writer.writerow([
            seed,
            config["gradient_clip"],
            clipping_stats["total_updates"],
            clipping_stats["clipped_updates"],
            clipping_stats["clipping_percentage"],
            clipping_stats["mean_gradient_norm"],
            clipping_stats["max_gradient_norm"]
        ])

    print(f"\nResults saved in: {output_dir}")

    print("\nGradient clipping statistics:")

    for key, value in clipping_stats.items():

        print(f"{key}: {value}")


# ============================================================
# 7. RUN TARGET UPDATE INTERVAL EXPERIMENTS
# ============================================================

def run_target_update_experiments():

    print("\nTARGET UPDATE INTERVAL SENSITIVITY EXPERIMENT")
    print("=============================================")

    print("Baseline target update interval: 1000")
    print("Experimental target update interval: 500")
    print("Learning rate: 0.0005")
    print("Training steps per seed: 100000")
    print("Random seeds:", EXPERIMENT_CONFIG["seeds"])

    for seed in EXPERIMENT_CONFIG["seeds"]:

        print(
            f"\nStarting target update experiment: Seed {seed}"
        )

        (
            rewards,
            losses,
            diagnostics,
            clipping_stats,
            agent
        ) = train_one_run(
            EXPERIMENT_CONFIG,
            seed
        )

        save_run_results(
            EXPERIMENT_CONFIG,
            seed,
            rewards,
            losses,
            diagnostics,
            clipping_stats,
            agent
        )

        print(
            f"Seed {seed} completed. "
            f"Episodes: {len(rewards)}"
        )

    print("\nAll target update interval experiments completed!")


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    run_target_update_experiments()
