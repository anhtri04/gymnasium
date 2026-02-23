# test_random_quick.py - Quick test without rendering
import numpy as np
from football_env import FootballEnv

def test_random_actions_quick(num_episodes=3, max_steps=100):
    env = FootballEnv()  # No render mode for quick test
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        print(f"Episode {episode + 1}")
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                print(f"  Ended at step {step}, reward: {episode_reward}, score: {info['score1']}-{info['score2']}")
                break
        else:
            print(f"  Completed {max_steps} steps, reward: {episode_reward}")
    env.close()
    print("\nRandom action test passed!")

if __name__ == "__main__":
    test_random_actions_quick()
