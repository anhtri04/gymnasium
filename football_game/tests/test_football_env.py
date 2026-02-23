# tests/test_football_env.py
import pytest
import gymnasium as gym

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
