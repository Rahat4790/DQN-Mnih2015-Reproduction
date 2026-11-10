
import torch

from src.network import QNetwork
from src.environment import make_env, get_env_info


# Create environment
env = make_env(seed=42)

# Obtain dimensions
state_dim, action_dim = get_env_info(env)

# Create neural network
q_network = QNetwork(state_dim, action_dim)

# Reset environment
state, info = env.reset(seed=42)

# Convert state into PyTorch tensor
state_tensor = torch.tensor(
    state,
    dtype=torch.float32
)

# Predict Q-values
q_values = q_network(state_tensor)

print("State:", state)
print("State shape:", state_tensor.shape)
print("Predicted Q-values:", q_values)
print("Q-values shape:", q_values.shape)

env.close()
