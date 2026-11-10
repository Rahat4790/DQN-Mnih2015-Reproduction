
from src.environment import make_env
from src.replay_buffer import ReplayBuffer


# Create environment
env = make_env(seed=42)

# Create replay buffer
buffer = ReplayBuffer(capacity=1000)

# Collect 10 experiences
state, info = env.reset(seed=42)

for i in range(10):

    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    buffer.add(
        state,
        action,
        reward,
        next_state,
        done
    )

    state = next_state

    if done:
        state, info = env.reset()

# Check buffer
print("Buffer size:", len(buffer))

# Sample 4 experiences
batch = buffer.sample(batch_size=4)

states, actions, rewards, next_states, dones = batch

print("Sampled batch size:", len(states))
print("Actions:", actions)
print("Rewards:", rewards)
print("Done flags:", dones)

env.close()
