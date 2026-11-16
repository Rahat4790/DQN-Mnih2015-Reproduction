
import random
import numpy as np
import torch
import csv
import json
import os

from tqdm import tqdm

from src.environment import make_env, get_env_info
from src.ablation_agent import AblationDQNAgent


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
# 2. BASELINE CONFIGURATION
# ============================================================

CONFIG = {

    "environment": "CartPole-v1",
    "total_steps": 100000,

    "buffer_capacity": 50000,
    "batch_size": 64,

    "gamma": 0.99,
    "learning_rate": 0.0005,

    "target_update_interval": 1000,

    "hidden_dims": (24, 24),

    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 20000,

    "gradient_clip": 10.0,

    "seeds": [42, 123, 2025, 7, 21]
}


ABLATION_MODES = [
    "no_replay"
]


# ============================================================
# 3. EPSILON SCHEDULE
# ============================================================

def get_epsilon(step):

    fraction = min(
        step / CONFIG["epsilon_decay_steps"],
        1.0
    )

    return (
        CONFIG["epsilon_start"]
        + fraction * (
            CONFIG["epsilon_end"]
            - CONFIG["epsilon_start"]
        )
    )


# ============================================================
# 4. TRAIN ONE RUN
# ============================================================

def train_one_run(ablation_mode, seed):

    set_seed(seed)

    env = make_env(seed=seed)

    state_dim, action_dim = get_env_info(env)

    agent = AblationDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        ablation_mode=ablation_mode,
        buffer_capacity=CONFIG["buffer_capacity"],
        batch_size=CONFIG["batch_size"],
        gamma=CONFIG["gamma"],
        learning_rate=CONFIG["learning_rate"],
        target_update_interval=CONFIG["target_update_interval"],
        hidden_dims=CONFIG["hidden_dims"],
        gradient_clip=CONFIG["gradient_clip"]
    )

    episode_rewards = []
    episode_losses = []
    episode_losses_current = []

    total_steps = 0

    state, info = env.reset(seed=seed)

    episode_reward = 0

    progress = tqdm(
        total=CONFIG["total_steps"],
        desc=f"{ablation_mode} | Seed {seed}"
    )

    while total_steps < CONFIG["total_steps"]:

        epsilon = get_epsilon(total_steps)

        action = agent.select_action(state, epsilon)

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

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

            state, info = env.reset()

            episode_reward = 0
            episode_losses_current = []

    progress.close()
    env.close()

    return episode_rewards, episode_losses, agent


# ============================================================
# 5. SAVE RESULTS
# ============================================================

def save_results(
    ablation_mode,
    seed,
    rewards,
    losses,
    agent
):

    output_dir = os.path.join(
        "results",
        "ablation",
        ablation_mode,
        f"seed_{seed}"
    )

    os.makedirs(output_dir, exist_ok=True)

    config_to_save = CONFIG.copy()

    config_to_save["hidden_dims"] = list(
        config_to_save["hidden_dims"]
    )

    config_to_save["ablation_mode"] = ablation_mode

    with open(
        os.path.join(output_dir, "config.json"),
        "w"
    ) as file:

        json.dump(config_to_save, file, indent=4)

    torch.save(
        agent.online_network.state_dict(),
        os.path.join(output_dir, "online_network.pth")
    )

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

    print(f"\nSaved: {output_dir}")


# ============================================================
# 6. RUN ALL ABLATION EXPERIMENTS
# ============================================================

def run_ablation_experiments():

    for ablation_mode in ABLATION_MODES:

        print("\n" + "=" * 55)
        print(f"STARTING ABLATION: {ablation_mode}")
        print("=" * 55)

        for seed in CONFIG["seeds"]:

            print(
                f"\nTraining {ablation_mode}, seed {seed}"
            )

            rewards, losses, agent = train_one_run(
                ablation_mode,
                seed
            )

            save_results(
                ablation_mode,
                seed,
                rewards,
                losses,
                agent
            )

            print(
                f"Completed {ablation_mode}, seed {seed}. "
                f"Episodes: {len(rewards)}"
            )

    print("\nALL ABLATION EXPERIMENTS COMPLETED!")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    run_ablation_experiments()
