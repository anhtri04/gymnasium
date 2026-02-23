# train_random.py
import numpy as np
from football_env import FootballEnv

def test_random_actions(num_episodes=5):
    env = FootballEnv(render_mode='human')
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        print(f"Episode {episode + 1}")
        for step in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                print(f"  Ended at step {step}, reward: {episode_reward}")
                break
    env.close()

if __name__ == "__main__":
    test_random_actions()
