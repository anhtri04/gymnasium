"""Train Stage 3 - Full game against opponent.

This is the final stage where the agent uses all learned skills:
- Ball control (from stages 1a-1c)
- Rotation and movement (from stage 1c)
- Shooting accuracy (from stages 2a-2c)

Now with an opponent to play against!

Usage:
    python train_full_game_3.py [model_path]     # Train
    python train_full_game_3.py eval [model_path] # Evaluate with rendering
"""
import yaml
import sys
import time
from pathlib import Path
from football_env import FootballEnv
from stable_baselines3 import PPO

# Load config
with open('configs/full_game_3.yaml', 'r') as f:
    config = yaml.safe_load(f)

def evaluate(model_path="models/full_game_3", num_episodes=5):
    """Evaluate trained agent with rendering."""
    print(f"\n{'='*60}")
    print("EVALUATION MODE - Watching agent play!")
    print(f"{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"Episodes to watch: {num_episodes}")
    print("\nPress Ctrl+C to stop early")
    print("="*60 + "\n")
    
    # Create environment WITH rendering
    env = FootballEnv(
        opponent_type=config['opponent']['type'],
        render_mode='human'
    )
    
    # Load model
    if not Path(model_path + ".zip").exists():
        print(f"❌ Model not found: {model_path}")
        env.close()
        return
    
    model = PPO.load(model_path, env=env)
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        try:
            while True:
                # Agent takes action
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # RENDER THE GAME!
                env.render()
                
                episode_reward += reward
                steps += 1
                
                # Slow down to watch (60 FPS)
                time.sleep(1/60)
                
                if terminated or truncated:
                    print(f"  Steps: {steps}")
                    print(f"  Score: {info['score1']}-{info['score2']}")
                    print(f"  Reward: {episode_reward:.2f}")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nEvaluation stopped by user")
            break
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")
    env.close()

def train():
    """Train the agent."""
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
    load_from = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != 'eval' else "models/shooting_2c"
    
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
    print(f"\nTo watch agent play, run:")
    print(f"  python train_full_game_3.py eval")
    
    env.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        # Evaluation mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else "models/full_game_3"
        evaluate(model_path)
    else:
        # Training mode
        train()
