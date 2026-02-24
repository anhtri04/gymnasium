"""
Run full curriculum training across multiple stages.
Progresses from simple skills to complex gameplay.
"""
import sys
from pathlib import Path

from configurable_env import ConfigurableFootballEnv
from stable_baselines3 import PPO
import yaml


def train_stage(config_path, timesteps=50000, load_from=None):
    """Train a single stage with optional weight transfer."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print(f"Stage: {config['stage_name']}")
    print(f"Mode: {config['mode']}")
    print(f"Timesteps: {timesteps:,}")
    if load_from:
        print(f"Loading from: {load_from}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = ConfigurableFootballEnv(config=config)
    
    # Load or create model
    if load_from and Path(load_from + ".zip").exists():
        print(f"Loading weights from {load_from}...")
        model = PPO.load(load_from, env=env)
    else:
        print("Creating new model...")
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
            clip_range=0.2
        )
    
    # Train
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Save
    model_name = f"models/stage_{config['stage']}"
    model.save(model_name)
    print(f"\nModel saved to {model_name}")
    
    env.close()
    return model_name


def main():
    """Run curriculum training."""
    # Define curriculum stages
    stages = [
        ("configs/ball_control_1a.yaml", 50000),
        ("configs/shooting_2a.yaml", 50000),
    ]
    
    previous_model = None
    
    for i, (config_path, timesteps) in enumerate(stages):
        print(f"\n{'#'*60}")
        print(f"CURRICULUM STEP {i+1}/{len(stages)}")
        print(f"{'#'*60}")
        
        # Train this stage
        model_path = train_stage(
            config_path=config_path,
            timesteps=timesteps,
            load_from=previous_model
        )
        
        # Save for next stage
        previous_model = model_path
    
    print(f"\n{'='*60}")
    print("CURRICULUM TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Final model: {previous_model}")


if __name__ == "__main__":
    main()
