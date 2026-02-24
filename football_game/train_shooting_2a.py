"""Train shooting stage 2a - wide goal, ball at feet."""
import yaml
from configurable_env import ConfigurableFootballEnv
from stable_baselines3 import PPO

# Load config
with open('configs/shooting_2a.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Training: {config['stage_name']}")
print(f"Mode: {config['mode']}")
print(f"Goal width: {config['goal']['width']}px")

# Create environment with config
env = ConfigurableFootballEnv(config=config)

# Check if we should load from previous stage
import sys
load_from = sys.argv[1] if len(sys.argv) > 1 else None

if load_from:
    print(f"\nLoading weights from: {load_from}")
    model = PPO.load(load_from, env=env)
else:
    print("\nCreating new model...")
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
model.save("models/shooting_2a")
print("\nModel saved to models/shooting_2a")

env.close()
