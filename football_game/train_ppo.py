"""
Train a single agent against dummy opponent using PPO.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from football_env import FootballEnv


def make_env():
    """Create environment."""
    return FootballEnv()


def train():
    """Train PPO agent."""
    print("Initializing environment...")
    env = make_env()
    
    print("Creating PPO model...")
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
    
    # Save checkpoints every 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./models/",
        name_prefix="ppo_football"
    )
    
    print("Starting training...")
    model.learn(
        total_timesteps=1_000_000,  # 1M steps
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("ppo_football_final")
    print("Training complete! Model saved as 'ppo_football_final'")
    
    env.close()


def evaluate():
    """Evaluate trained agent."""
    print("Loading trained model...")
    env = make_env()
    model = PPO.load("ppo_football_final")
    
    print("Running evaluation episodes...")
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, Steps={steps}, Score={info['score1']}-{info['score2']}")
                break
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        train()
