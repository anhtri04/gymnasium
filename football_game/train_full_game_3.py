"""Train Stage 3 - Full game against opponent.

This is the final stage where the agent uses all learned skills:
- Ball control (from stages 1a-1c)
- Rotation and movement (from stage 1c)
- Shooting accuracy (from stages 2a-2c)

Now with an opponent to play against!
"""
import yaml
import sys
from pathlib import Path
from football_env import FootballEnv  # Use original env for full game
from stable_baselines3 import PPO

# Load config
with open('configs/full_game_3.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Training: {config['stage_name']}")
print(f"Mode: {config['mode']}")
print(f"Opponent: {config['opponent']['type']}")
print(f"Field: {config['field']['width']}x{config['field']['height']}")
print("\n🎯 FINAL STAGE: Combining all learned skills!")
print("   - Ball control and chasing")
print("   - Rotation and positioning")  
print("   - Shooting accuracy")
print("   - Playing against opponent")

# Create environment with opponent
env = FootballEnv(
    opponent_type=config['opponent']['type'],
    render_mode=None
)

# Load from previous stage (shooting_2c)
load_from = sys.argv[1] if len(sys.argv) > 1 else "models/shooting_2c"

if load_from and Path(load_from + ".zip").exists():
    print(f"\nLoading weights from: {load_from}")
    print("Transferring skills from shooting stage...")
    model = PPO.load(load_from, env=env)
else:
    print("\n⚠️  No previous model found, starting fresh")
    print("   (Should load from shooting_2c for best results)")
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

print("\nStarting training for 100k steps...")
print("(Longer training for complex gameplay)")
model.learn(total_timesteps=100000, progress_bar=True)

# Save
model.save("models/full_game_3")
print(f"\n{'='*60}")
print("FINAL MODEL SAVED!")
print(f"{'='*60}")
print(f"Location: models/full_game_3")
print("\nAgent can now:")
print("  ✓ Control the ball")
print("  ✓ Rotate to face targets")
print("  ✓ Shoot accurately")
print("  ✓ Play against opponent")

env.close()
