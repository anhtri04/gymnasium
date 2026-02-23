# verify_env.py
import numpy as np
from football_env import FootballEnv

def verify_env():
    print("Verifying FootballEnv...")
    env = FootballEnv()
    
    # Check spaces
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run episodes
    for ep in range(3):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert reward in [-1.0, 0.0, 1.0]
            if terminated or truncated:
                print(f"Episode {ep} ended at step {step}")
                break
    
    print("All verifications passed!")
    env.close()

if __name__ == "__main__":
    verify_env()
