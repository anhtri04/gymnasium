"""Train ball control stage 1a - multiple balls close together."""
import yaml
from configurable_env import ConfigurableFootballEnv
from stable_baselines3 import PPO

# Load config
with open('configs/ball_control_1a.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Training: {config['stage_name']}")
print(f"Mode: {config['mode']}")
print(f"Field size: {config['field']['width']}x{config['field']['height']}")
print(f"Balls: {len(config['balls'])}")

# Create environment with config
env = ConfigurableFootballEnv(config=config)

# Create model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./tensorboard_logs/"
)

print("\nStarting training for 50k steps...")
model.learn(total_timesteps=50000, progress_bar=True)

# Save model
model.save("models/ball_control_1a")
print("\nModel saved to models/ball_control_1a")

env.close()
