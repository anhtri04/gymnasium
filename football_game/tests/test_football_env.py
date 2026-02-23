# tests/test_football_env.py
import pytest
import gymnasium as gym
import numpy as np

def test_env_registration():
    from football_env import FootballEnv
    env = FootballEnv()
    assert env is not None
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'action_space')

def test_env_inherits_gym_env():
    from football_env import FootballEnv
    import gymnasium as gym
    env = FootballEnv()
    assert isinstance(env, gym.Env)

def test_observation_shape():
    from football_env import FootballEnv
    env = FootballEnv()
    obs, _ = env.reset()
    assert obs.shape == (10,)
    assert obs.dtype == np.float32

def test_observation_normalization():
    from football_env import FootballEnv
    env = FootballEnv()
    obs, _ = env.reset()
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)

def test_observation_after_step():
    from football_env import FootballEnv
    env = FootballEnv()
    obs1, _ = env.reset()
    obs2, _, _, _, _ = env.step(0)
    assert not np.array_equal(obs1, obs2)

def test_step_returns_correct_tuple():
    from football_env import FootballEnv
    env = FootballEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (10,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_step_moves_player():
    from football_env import FootballEnv
    env = FootballEnv()
    env.reset()
    initial_x = env.state.player1.x
    for _ in range(10):
        env.step(0)
    assert env.state.player1.x != initial_x
