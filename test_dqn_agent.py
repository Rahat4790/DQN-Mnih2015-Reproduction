
from src.environment import make_env, get_env_info
from src.dqn_agent import DQNAgent


# Create environment
env = make_env(seed=42)

state_dim, action_dim = get_env_info(env)

# Create DQN agent
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    batch_size=4,
    target_update_interval=5
)

# Reset environment
state, info = env.reset(seed=42)

# Test action selection
action = agent.select_action(state, epsilon=0.0)

print("State:", state)
print("Selected action:", action)

# Collect experiences
for i in range(10):

    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = (
        env.step(action)
    )

    done = terminated or truncated

    agent.store_experience(
        state,
        action,
        reward,
        next_state,
        done
    )

    state = next_state

    if done:
        state, info = env.reset()

# Test learning
loss = agent.learn()

print("Replay buffer size:", len(agent.replay_buffer))
print("Learning steps:", agent.learning_steps)
print("TD loss:", loss)
print("Device:", agent.device)

env.close()
