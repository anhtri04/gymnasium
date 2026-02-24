# Phase 3 — Single Agent Baseline with Stable-Baselines3

> **Goal:** Train one agent against a fixed dummy opponent using PPO from stable-baselines3.

**Approach:** Use SB3's PPO implementation. We only need to:
1. Install stable-baselines3
2. Wrap our FootballEnv for SB3 (if needed)
3. Create a training script
4. Train against dummy opponent
5. Verify learning

**Tech Stack:** Python, stable-baselines3, torch, existing FootballEnv

---

## Task 1: Install Stable-Baselines3

**Step 1: Update requirements.txt**

```
pygame>=2.5.0
gymnasium>=0.29.0
numpy>=1.24.0
stable-baselines3>=2.0.0
torch>=2.0.0
```

**Step 2: Install**

```bash
pip install -r requirements.txt
```

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add stable-baselines3 and torch for PPO training"
```

---

## Task 2: Create Training Script

**Create: `train_ppo.py`**

```python
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
```

**Step 4: Test import**

```bash
python -c "from stable_baselines3 import PPO; print('SB3 imported successfully')"
```

**Step 5: Commit**

```bash
git add train_ppo.py
git commit -m "feat: add PPO training script with SB3"
```

---

## Task 3: Add Dummy Opponent

**Modify: `football_env.py`**

Add opponent parameter to __init__:

```python
def __init__(self, render_mode=None, opponent_type="stationary"):
    # ... existing code ...
    self.opponent_type = opponent_type
```

Update step() to move opponent:

```python
def step(self, action):
    # ... existing action execution for player 1 ...
    
    # Move opponent (player 2) based on opponent_type
    if self.opponent_type == "random":
        # Random action for player 2
        opponent_action = self.action_space.sample()
        self._execute_action_for_player(opponent_action, player=2)
    elif self.opponent_type == "stationary":
        # Do nothing
        pass
    
    # ... rest of step logic ...
```

**Step 4: Commit**

```bash
git add football_env.py
git commit -m "feat: add dummy opponent (stationary/random)"
```

---

## Task 4: Quick Training Test

**Create: `train_quick.py`**

```python
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
```

**Step 5: Run test**

```bash
python train_quick.py
```

Expected: Training progresses, agent learns basic movement

**Step 6: Commit**

```bash
git add train_quick.py
git commit -m "feat: add quick training test"
```

---

## Task 5: Full Training

**Step 1: Start training**

```bash
python train_ppo.py
```

This will:
- Train for 1M timesteps
- Save checkpoints every 100k steps
- Log to tensorboard
- Take ~10-30 minutes depending on hardware

**Step 2: Monitor progress**

Training stats will show:
- Episode reward mean (should increase over time)
- Approx KL (policy update magnitude)
- Loss values

**Step 3: Evaluate**

```bash
python train_ppo.py eval
```

Should show positive rewards if agent learned to score.

---

## Summary

**Phase 3 uses stable-baselines3 which provides:**
- PPO algorithm (proven effective for continuous control)
- MlpPolicy (neural network: observation -> action probabilities)
- Value function (critic for advantage estimation)
- Experience buffer (rollout collection)
- Clipped surrogate objective (PPO update)
- All hyperparameters tuned for common use cases

**We focus on:**
- Clean environment interface
- Proper reward signals
- Opponent logic
- Training loop orchestration

**Next:** Phase 4 (Two Agent Self-Play) after verifying single agent works.
