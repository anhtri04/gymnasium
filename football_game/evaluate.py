"""Evaluate trained agent - Watch it play!

Usage:
    python evaluate.py [stage] [--episodes N] [--model PATH]
    
Examples:
    python evaluate.py 1a                 # Evaluate stage 1a
    python evaluate.py 3 --episodes 10    # Evaluate stage 3, 10 episodes
    python evaluate.py --model models/custom_model --episodes 5
"""
import sys
import time
import argparse
from pathlib import Path
from football_env import FootballEnv
from configurable_env import ConfigurableFootballEnv
from stable_baselines3 import PPO
import yaml


def load_config_for_stage(stage):
    """Load config file for given stage."""
    config_map = {
        '1a': 'configs/ball_control_1a.yaml',
        '1b': 'configs/ball_control_1b.yaml',
        '1c': 'configs/ball_control_1c.yaml',
        '2a': 'configs/shooting_2a.yaml',
        '2b': 'configs/shooting_2b.yaml',
        '2c': 'configs/shooting_2c.yaml',
        '3': 'configs/full_game_3.yaml'
    }
    
    if stage not in config_map:
        print(f"❌ Unknown stage: {stage}")
        print(f"Available stages: {', '.join(config_map.keys())}")
        return None
    
    config_path = config_map[stage]
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_agent(model_path, config=None, num_episodes=5, delay=1/60):
    """Evaluate agent with rendering."""
    print(f"\n{'='*70}")
    print("🎮 AGENT EVALUATION - Watching it play!")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render speed: {1/delay:.0f} FPS")
    print(f"\nPress Ctrl+C to stop early")
    print("="*70 + "\n")
    
    # Check if model exists
    if not Path(model_path + ".zip").exists():
        print(f"❌ Model not found: {model_path}")
        print("\nTrain a model first:")
        print(f"  python train_ball_control_1a.py")
        return
    
    # Create environment based on config
    if config:
        print(f"Stage: {config.get('stage_name', 'Unknown')}")
        print(f"Mode: {config['mode']}\n")
        env = ConfigurableFootballEnv(config=config, render_mode='human')
    else:
        # Default to full game
        env = FootballEnv(opponent_type='stationary', render_mode='human')
    
    # Load model
    print("Loading model...")
    model = PPO.load(model_path, env=env)
    print("✅ Model loaded!\n")
    
    # Run episodes
    total_reward = 0
    total_steps = 0
    wins = 0
    
    for episode in range(num_episodes):
        print(f"--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        try:
            while True:
                # Agent takes action
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # RENDER THE GAME! (This shows the pygame window)
                env.render()
                
                episode_reward += reward
                steps += 1
                
                # Slow down to watch
                time.sleep(delay)
                
                if terminated or truncated:
                    score1 = info.get('score1', 0)
                    score2 = info.get('score2', 0)
                    win = score1 > score2
                    
                    print(f"  Steps: {steps:4d} | Score: {score1}-{score2} | Reward: {episode_reward:7.2f} | {'✅ WIN' if win else '❌ LOSS'}")
                    
                    total_reward += episode_reward
                    total_steps += steps
                    if win:
                        wins += 1
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Evaluation stopped by user")
            break
    
    # Summary
    episodes_run = episode + 1
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Episodes: {episodes_run}")
    print(f"Wins: {wins}/{episodes_run} ({wins/episodes_run*100:.1f}%)")
    print(f"Avg Steps: {total_steps/episodes_run:.1f}")
    print(f"Avg Reward: {total_reward/episodes_run:.2f}")
    print(f"{'='*70}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained football agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py 1a                    # Evaluate stage 1a
  python evaluate.py 3 --episodes 10       # Evaluate stage 3, 10 episodes
  python evaluate.py --model models/custom # Evaluate specific model
  python evaluate.py 2c --delay 0.05       # Slower speed (20 FPS)
        """
    )
    
    parser.add_argument('stage', nargs='?', 
                       help='Stage to evaluate (1a, 1b, 1c, 2a, 2b, 2c, 3)')
    parser.add_argument('--model', '-m', 
                       help='Path to specific model (overrides stage)')
    parser.add_argument('--episodes', '-e', type=int, default=5,
                       help='Number of episodes to run (default: 5)')
    parser.add_argument('--delay', '-d', type=float, default=1/60,
                       help='Delay between frames in seconds (default: 1/60 = 60 FPS)')
    
    args = parser.parse_args()
    
    # Determine model path and config
    if args.model:
        model_path = args.model
        config = None
    elif args.stage:
        config = load_config_for_stage(args.stage)
        if config is None:
            return
        model_path = f"models/stage_{args.stage}"
    else:
        print("❌ Please specify a stage (1a, 1b, etc.) or use --model")
        parser.print_help()
        return
    
    # Run evaluation
    evaluate_agent(model_path, config, args.episodes, args.delay)


if __name__ == "__main__":
    main()
