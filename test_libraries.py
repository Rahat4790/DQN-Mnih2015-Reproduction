import torch
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib
import tqdm

print("===== LIBRARY CHECK =====")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("Gymnasium:", gym.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)

print("\n===== DEVICE CHECK =====")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)

print("\n===== CARTPOLE CHECK =====")

env = gym.make("CartPole-v1")

state, info = env.reset(seed=42)

print("Initial state:", state)
print("State dimension:", env.observation_space.shape)
print("Action space:", env.action_space.n)

env.close()

print("\nAll checks completed!")