# Phase 2 — Gymnasium Environment Wrapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the 2D football game into a standardized `gymnasium.Env` that AI agents can train on, with proper observation/action spaces, step/reset mechanics, and reward signals.

**Architecture:** Single-player perspective where agent controls Player 1 (attacking right). The environment wraps the existing game engine and exposes it through the Gymnasium API. Observations are normalized to [0,1] or [-1,1] ranges for stable neural network training.

**Tech Stack:** Python, gymnasium, numpy, existing game engine (state.py, physics.py, entities.py)

---

## Project Structure Updates

```
football_game/
├── requirements.txt         # Add gymnasium
├── football_env.py          # NEW: Gymnasium environment wrapper
├── train_random.py          # NEW: Test script with random actions
├── main.py                  # Existing (manual play)
├── config.py                # Existing
├── state.py                 # Existing
├── entities.py              # Existing
├── physics.py               # Existing
├── renderer.py              # Existing
└── tests/
    ├── test_football_env.py # NEW: Tests for gym environment
    └── ... (existing tests)
```

---

## Part A: Setup Gymnasium Dependency

### Task 0: Update Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add gymnasium to requirements**

```
pygame>=2.5.0
gymnasium>=0.29.0
numpy>=1.24.0
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: Successfully installed gymnasium, numpy

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add gymnasium and numpy for RL environment"
```

---

## Part B: Core Environment Structure

### Task 1: Create FootballEnv Class Skeleton

**Files:**
- Create: `football_env.py`
- Create: `tests/test_football_env.py`

**Step 1: Write test**

```python
# tests/test_football_env.py
import pytest
import gymnasium as gym

def test_env_registration():
    from football_env import FootballEnv
    env = FootballEnv()
    assert env is not None
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'action_space')

def test_env_inherits_gym_env():
    from football_env import FootballEnv
    import gymnasium as gym
    env = FootballEnv()
    assert isinstance(env, gym.Env)
```

**Step 2: Run test (should fail)**

Run: `pytest tests/test_football_env.py::test_env_registration -v`
Expected: FAIL - FootballEnv not defined

**Step 3: Implement FootballEnv skeleton**

```python
# football_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from state import GameState
from physics import (
    check_ball_wall_collision,
    check_player_wall_collision,
    check_goal,
    is_ball_in_kick_arc,
    kick_ball,
    dribble_ball
)
from config import (
    FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT,
    SCREEN_WIDTH, SCREEN_HEIGHT,
    PLAYER_SPEED, PLAYER_ROTATION_SPEED,
    FPS, EPISODE_TIME_LIMIT
)


class FootballEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        self.state = GameState()
        
        # Observation space: 10 continuous values normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(10,),
            dtype=np.float32
        )
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # For rendering
        self.screen = None
        self.clock = None
        
    def _get_obs(self) -> np.ndarray:
        raise NotImplementedError("Will implement in Task 2")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError("Will implement in Task 3")
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        raise NotImplementedError("Will implement in Task 4")
    
    def render(self):
        raise NotImplementedError("Will implement in Task 6")
    
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
```

**Step 4: Run test (should pass)**

Run: `pytest tests/test_football_env.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add football_env.py tests/test_football_env.py
git commit -m "feat: add FootballEnv skeleton class with gymnasium.Env inheritance"
```

---

### Task 2: Implement Observation Space

**Files:**
- Modify: `football_env.py`

**Step 1: Write test**

```python
# tests/test_football_env.py
import numpy as np

def test_observation_shape():
    from football_env import FootballEnv
    env = FootballEnv()
    obs, _ = env.reset()
    assert obs.shape == (10,)
    assert obs.dtype == np.float32

def test_observation_normalization():
    from football_env import FootballEnv
    env = FootballEnv()
    obs, _ = env.reset()
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)

def test_observation_after_step():
    from football_env import FootballEnv
    env = FootballEnv()
    obs1, _ = env.reset()
    obs2, _, _, _, _ = env.step(0)
    assert not np.array_equal(obs1, obs2)
```

**Step 2: Run test (should fail)**

Run: `pytest tests/test_football_env.py::test_observation_shape -v`
Expected: FAIL - _get_obs returns NotImplementedError

**Step 3: Implement _get_obs**

```python
def _get_obs(self) -> np.ndarray:
    # Normalize positions to [0, 1]
    p1_x_norm = self.state.player1.x / SCREEN_WIDTH
    p1_y_norm = self.state.player1.y / SCREEN_HEIGHT
    p2_x_norm = self.state.player2.x / SCREEN_WIDTH
    p2_y_norm = self.state.player2.y / SCREEN_HEIGHT
    ball_x_norm = self.state.ball.x / SCREEN_WIDTH
    ball_y_norm = self.state.ball.y / SCREEN_HEIGHT
    
    # Normalize angles to [0, 1]
    p1_angle_norm = self.state.player1.angle / 360.0
    p2_angle_norm = self.state.player2.angle / 360.0
    
    # Normalize velocities to [0, 1]
    max_speed = 20.0
    ball_vx_norm = (self.state.ball.vx / max_speed + 1) / 2
    ball_vy_norm = (self.state.ball.vy / max_speed + 1) / 2
    
    obs = np.array([
        p1_x_norm, p1_y_norm, p1_angle_norm,
        p2_x_norm, p2_y_norm, p2_angle_norm,
        ball_x_norm, ball_y_norm, ball_vx_norm, ball_vy_norm
    ], dtype=np.float32)
    
    return np.clip(obs, 0.0, 1.0)
```

**Step 4: Run test (should pass)**

Run: `pytest tests/test_football_env.py::test_observation_shape -v`
Expected: PASS

**Step 5: Commit**

```bash
git add football_env.py tests/test_football_env.py
git commit -m "feat: implement observation space with 10 normalized values"
```

---

### Task 3: Implement reset()

**Files:**
- Modify: `football_env.py`

**Step 1: Write test**

```python
# tests/test_football_env.py
def test_reset_returns_observation():
    from football_env import FootballEnv
    env = FootballEnv()
    obs, info = env.reset()
    assert obs.shape == (10,)
    assert isinstance(info, dict)

def test_reset_clears_episode_time():
    from football_env import FootballEnv
    from config import FPS
    env = FootballEnv()
    env.reset()
    for _ in range(60 * FPS):
        env.step(0)
    assert env.state.episode_time > 0
    env.reset()
    assert env.state.episode_time == 0.0
```

**Step 2: Run test (should fail)**

Run: `pytest tests/test_football_env.py::test_reset_returns_observation -v`
Expected: FAIL - reset() raises NotImplementedError

**Step 3: Implement reset**

```python
def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
    super().reset(seed=seed)
    self.state.reset_positions()
    info = {'episode_time': 0.0, 'score1': 0, 'score2': 0}
    return self._get_obs(), info
```

**Step 4: Run test (should pass)**

Run: `pytest tests/test_football_env.py::test_reset_returns_observation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add football_env.py tests/test_football_env.py
git commit -m "feat: implement reset() with initial observation and info"
```

---

### Task 4: Implement step()

**Files:**
- Modify: `football_env.py`

**Step 1: Write test**

```python
# tests/test_football_env.py
def test_step_returns_correct_tuple():
    from football_env import FootballEnv
    env = FootballEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (10,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_step_moves_player():
    from football_env import FootballEnv
    env = FootballEnv()
    env.reset()
    initial_x = env.state.player1.x
    for _ in range(10):
        env.step(0)
    assert env.state.player1.x != initial_x
```

**Step 2: Run test (should fail)**

Run: `pytest tests/test_football_env.py::test_step_returns_correct_tuple -v`
Expected: FAIL - step() raises NotImplementedError

**Step 3: Implement step**

```python
def step(self, action: int):
    assert self.action_space.contains(action)
    
    # Execute action
    if action == 0:
        self.state.player1.move_forward(PLAYER_SPEED)
        dribble_ball(self.state.player1, self.state.ball)
    elif action == 1:
        self.state.player1.move_backward(PLAYER_SPEED)
    elif action == 2:
        self.state.player1.rotate(PLAYER_ROTATION_SPEED)
    elif action == 3:
        self.state.player1.rotate(-PLAYER_ROTATION_SPEED)
    elif action == 4:
        if is_ball_in_kick_arc(self.state.player1, self.state.ball):
            kick_ball(self.state.player1, self.state.ball)
    
    # Update physics
    self.state.ball.update()
    check_ball_wall_collision(self.state.ball)
    check_player_wall_collision(self.state.player2)
    
    # Calculate reward and termination
    goal = check_goal(self.state.ball)
    reward = 0.0
    terminated = False
    
    if goal == "right":
        self.state.increment_score(1)
        reward = 1.0
        terminated = True
    elif goal == "left":
        self.state.increment_score(2)
        reward = -1.0
        terminated = True
    
    # Check time limit
    dt = 1.0 / FPS
    truncated = self.state.update_episode_time(dt)
    
    info = {
        'episode_time': self.state.episode_time,
        'score1': self.state.score1,
        'score2': self.state.score2,
        'ball_in_kick_arc': is_ball_in_kick_arc(self.state.player1, self.state.ball)
    }
    
    return self._get_obs(), reward, terminated, truncated, info
```

**Step 4: Run test (should pass)**

Run: `pytest tests/test_football_env.py::test_step_returns_correct_tuple -v`
Expected: PASS

**Step 5: Commit**

```bash
git add football_env.py tests/test_football_env.py
git commit -m "feat: implement step() with action execution and reward calculation"
```

---

### Task 5: Test Random Action Play

**Files:**
- Create: `train_random.py`

**Step 1: Write random agent test script**

```python
# train_random.py
import numpy as np
from football_env import FootballEnv

def test_random_actions(num_episodes=5):
    env = FootballEnv(render_mode='human')
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        print(f"Episode {episode + 1}")
        for step in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                print(f"  Ended at step {step}, reward: {episode_reward}")
                break
    env.close()

if __name__ == "__main__":
    test_random_actions()
```

**Step 2: Run test**

Run: `python train_random.py`
Expected: Environment runs with random actions

**Step 3: Commit**

```bash
git add train_random.py
git commit -m "feat: add random action test script"
```

---

### Task 6: Implement render()

**Files:**
- Modify: `football_env.py`

**Step 1: Write test**

```python
# tests/test_football_env.py
def test_render_no_crash():
    from football_env import FootballEnv
    env = FootballEnv(render_mode='human')
    env.reset()
    env.render()
    env.close()
```

**Step 2: Run test (should fail)**

Run: `pytest tests/test_football_env.py::test_render_no_crash -v`
Expected: FAIL - render() raises NotImplementedError

**Step 3: Implement render**

```python
def render(self):
    if self.render_mode is None:
        return None
    
    if self.render_mode == 'human':
        import pygame
        from renderer import render_field, render_player, render_ball, render_scoreboard
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("2D Football Game")
            self.clock = pygame.time.Clock()
        
        render_field(self.screen)
        render_ball(self.screen, self.state.ball)
        render_player(self.screen, self.state.player1)
        render_player(self.screen, self.state.player2)
        render_scoreboard(self.screen, self.state.score1, self.state.score2)
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(FPS)
    
    elif self.render_mode == 'rgb_array':
        import pygame
        from renderer import render_field, render_player, render_ball, render_scoreboard
        
        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        render_field(surface)
        render_ball(surface, self.state.ball)
        render_player(surface, self.state.player1)
        render_player(surface, self.state.player2)
        render_scoreboard(surface, self.state.score1, self.state.score2)
        
        frame = pygame.surfarray.array3d(surface)
        return np.transpose(frame, (1, 0, 2))
```

**Step 4: Run test (should pass)**

Run: `pytest tests/test_football_env.py::test_render_no_crash -v`
Expected: PASS

**Step 5: Commit**

```bash
git add football_env.py tests/test_football_env.py
git commit -m "feat: add render() with human and rgb_array modes"
```

---

### Task 7: Verify Environment

**Files:**
- Create: `verify_env.py`

**Step 1: Write verification script**

```python
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
```

**Step 2: Run verification**

Run: `python verify_env.py`
Expected: All checks pass

**Step 3: Commit**

```bash
git add verify_env.py
git commit -m "feat: add environment verification script"
```

---

## Summary

**Phase 2 Complete!** You now have:

1. **Gymnasium Environment** (`football_env.py`)
   - Proper observation space (10 normalized values)
   - Action space (5 discrete actions)
   - `reset()` and `step()` methods
   - Reward structure (+1 score, -1 concede)
   - Rendering support

2. **Tests** (`tests/test_football_env.py`)
3. **Verification Scripts** (`train_random.py`, `verify_env.py`)

**Next Steps:**
- Run `python -m pytest tests/test_football_env.py -v`
- Run `python verify_env.py`
- Proceed to Phase 3: Single Agent Baseline (PPO training)
