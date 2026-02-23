"""Quick training test (10k steps) to verify setup works."""
from football_env import FootballEnv
from stable_baselines3 import PPO

env = FootballEnv(opponent_type="stationary")
model = PPO("MlpPolicy", env, verbose=1)

print("Training for 10k steps...")
model.learn(total_timesteps=10_000)

print("Testing trained agent...")
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        print(f"Episode ended, reward: {reward}")
        break

env.close()
print("Quick test complete!")
