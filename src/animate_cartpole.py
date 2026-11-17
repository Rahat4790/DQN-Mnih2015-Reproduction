
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from src.environment import make_env, get_env_info
from src.network import QNetwork


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
EVALUATION_SEED = 1000

MODEL_PATH = Path(
    "results/sensitivity/learning_rate/lr_0.001/"
    "seed_42/online_network.pth"
)

OUTPUT_DIR = Path("results/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_GIF = OUTPUT_DIR / "cartpole_trained_dqn_seed42.gif"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model(env):

    state_dim, action_dim = get_env_info(env)

    model = QNetwork(
        state_dim,
        action_dim,
        hidden_dims=(24, 24)
    ).to(DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# ============================================================
# MAIN ANIMATION
# ============================================================

def main():

    print("\nDQN CARTPOLE DEMONSTRATION")
    print("=========================")

    env = make_env(
        seed=EVALUATION_SEED,
        render_mode="rgb_array"
    )

    model = load_model(env)

    state, info = env.reset(seed=EVALUATION_SEED)

    total_reward = 0
    frames = []
    done = False
    step = 0

    while not done:

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax(dim=1).item()

        state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward
        step += 1

        # Capture the rendered environment
        frame = env.render()

        if frame is not None:
            image = Image.fromarray(frame).convert("RGB")

            draw = ImageDraw.Draw(image)

            draw.rectangle(
                (5, 5, 270, 75),
                fill="white"
            )

            draw.text(
                (15, 12),
                f"DQN CartPole | Seed {SEED}",
                fill="black"
            )

            draw.text(
                (15, 32),
                f"Reward: {int(total_reward)}",
                fill="black"
            )

            draw.text(
                (15, 52),
                f"Step: {step}",
                fill="black"
            )

            # Keep every second frame to reduce GIF size
            if step % 2 == 0 or done:
                frames.append(image)

    env.close()

    if not frames:
        raise RuntimeError("No animation frames were captured.")

    # Save GIF
    frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        optimize=True
    )

    print("\n========== RESULTS ==========")
    print(f"Training seed: {SEED}")
    print(f"Learning rate: 0.001")
    print(f"Evaluation seed: {EVALUATION_SEED}")
    print(f"Episode reward: {total_reward}")
    print(f"Episode steps: {step}")
    print(f"GIF saved: {OUTPUT_GIF}")

    if total_reward == 500:
        print("This episode reached the CartPole maximum reward.")
    else:
        print("The episode ended before reaching the maximum reward.")

    print("=============================")


if __name__ == "__main__":
    main()
