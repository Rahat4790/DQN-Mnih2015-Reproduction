
import gymnasium as gym


def make_env(seed=None, render_mode=None):
    """
    Create the CartPole-v1 environment.

    Args:
        seed: Random seed for reproducibility.
        render_mode: Optional rendering mode.

    Returns:
        A Gymnasium environment.
    """

    env = gym.make(
        "CartPole-v1",
        render_mode=render_mode
    )

    if seed is not None:
        env.action_space.seed(seed)

    return env


def get_env_info(env):
    """
    Extract the state dimension and number of actions.
    """

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    return state_dim, action_dim
