# 2D Football RL - Gymnasium Environment

A complete 2D top-down football (soccer) game engine built with pygame, featuring reinforcement learning training through Stable-Baselines3 and curriculum learning.

## Overview

This project implements a 1v1 football game where AI agents learn to play through reinforcement learning. The game features physics-driven ball mechanics, rectangular rotatable players, and a comprehensive curriculum learning system that progressively trains agents from basic ball control to full competitive gameplay.

## Features

### Game Engine
- **Physics-driven ball** with friction, bouncing, and velocity
- **Rotatable rectangular players** with direction-based movement
- **Kick arc mechanics** for realistic ball kicking (45° cone in front of player)
- **Dribbling system** when moving with the ball
- **Goal detection and scoring** with automatic reset
- **Wall collision detection** for both players and ball
- **Time-limited matches** (60 seconds per episode)
- **Scoreboard display** with real-time scoring

### Reinforcement Learning
- **Gymnasium environment** (`FootballEnv`) with standardized API
- **Stable-Baselines3 integration** for PPO training
- **Configurable environment** for curriculum learning
- **Reward shaping** with multiple reward types:
  - Ball touch rewards
  - Kick execution rewards
  - Ball progress toward goal
  - Proximity bonuses
  - Goal/concede sparse rewards

### Curriculum Learning
7-stage progressive training:

1. **Ball Control 1a-c**: Learn to approach and control the ball
2. **Shooting 2a-c**: Master shooting accuracy from various positions
3. **Full Game 3**: Complete competitive play with opponent

## Installation

```bash
# Clone or navigate to project
cd gymnasium/football_game

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- pygame >= 2.5.0
- gymnasium >= 0.29.0
- stable-baselines3 >= 2.0.0
- torch >= 2.0.0
- numpy >= 1.24.0

## Project Structure

```
gymnasium/
├── football_game/
│   ├── main.py                    # Manual play entry point
│   ├── football_env.py            # Gymnasium environment
│   ├── configurable_env.py        # Curriculum learning wrapper
│   ├── train_ppo.py              # PPO training script
│   ├── run_curriculum.py         # Full curriculum training
│   ├── evaluate.py               # Model evaluation
│   ├── config.py                 # Game constants
│   ├── state.py                  # GameState management
│   ├── entities.py               # Player and Ball classes
│   ├── physics.py                # Collision and physics
│   ├── renderer.py               # Pygame rendering
│   ├── controls.py               # Keyboard input handling
│   ├── requirements.txt          # Dependencies
│   ├── configs/                  # Curriculum configurations
│   │   ├── ball_control_1a.yaml
│   │   ├── ball_control_1b.yaml
│   │   ├── ball_control_1c.yaml
│   │   ├── shooting_2a.yaml
│   │   ├── shooting_2b.yaml
│   │   ├── shooting_2c.yaml
│   │   └── full_game_3.yaml
│   ├── models/                   # Saved model checkpoints
│   └── tests/                    # Test suite
│       ├── test_state.py
│       ├── test_physics.py
│       ├── test_entities.py
│       └── test_renderer.py
└── docs/plans/                   # Implementation plans
    ├── 2025-02-23-phase-1-game-engine.md
    ├── 2025-02-23-phase-2-gym-env.md
    └── 2025-02-23-phase-3-sb3-training.md
```

## Usage

### Manual Play
Play the game with keyboard controls:

```bash
python main.py
```

**Controls:**
- **Player 1 (Red - Left):**
  - `W/S` - Move forward/backward
  - `A/D` - Rotate left/right
  - `SPACE` - Kick ball

- **Player 2 (Blue - Right):**
  - `↑/↓` - Move forward/backward
  - `←/→` - Rotate left/right
  - `ENTER` - Kick ball

- **General:**
  - `ESC` - Quit

### Training

#### Quick Test (10k steps)
```bash
python train_quick.py
```

#### Full PPO Training (1M steps)
```bash
python train_ppo.py
```

#### Curriculum Learning
Train through all 7 stages with weight transfer:

```bash
python run_curriculum.py
```

This will:
- Train Stage 1a-c: Ball control (50k steps each)
- Train Stage 2a-c: Shooting accuracy (50k steps each)
- Train Stage 3: Full game (100k steps)
- Transfer weights between stages
- Save models to `models/stage_*.zip`

#### Single Stage Training
Train a specific curriculum stage:

```bash
python train_ball_control_1a.py
python train_shooting_2b.py
python train_full_game_3.py
```

### Evaluation

```bash
# Evaluate final model
python train_ppo.py eval

# Evaluate with rendering
python evaluate.py --model models/stage_3.zip --render
```

### TensorBoard Monitoring
View training metrics:

```bash
tensorboard --logdir ./tensorboard_logs/
```

## Environment API

### Observation Space
10 continuous values normalized to [0, 1]:
- Player 1: x, y, angle (3 values)
- Player 2: x, y, angle (3 values)
- Ball: x, y, vx, vy (4 values)

### Action Space
5 discrete actions:
- `0` - Move forward
- `1` - Move backward
- `2` - Rotate left
- `3` - Rotate right
- `4` - Kick ball

### Reward Structure
- **Goal scored**: +1.0
- **Goal conceded**: -1.0
- **Ball touch**: +0.01
- **Kick executed**: +0.05
- **Ball progress toward goal**: +0.1 × progress
- **Entering opponent half**: +0.05
- **Proximity to goal**: Up to +0.4 based on distance

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_physics.py -v

# With coverage
pytest tests/ --cov=.
```

## Configuration

Curriculum stages are configured via YAML files in `configs/`. Each config specifies:
- Training mode (ball_control, shooting, full_game)
- Field dimensions and positions
- Ball and agent starting positions
- Reward weights
- Opponent type

Example:
```yaml
mode: shooting
stage: "2a"
stage_name: "Shooting - Wide Goal, Ball at Feet"

field:
  width: 400
  height: 300
  start_x: 400
  start_y: 250

balls:
  - at_feet: true

agent:
  x: 500
  y: 400
  angle: 0

opponent:
  type: "stationary"
```

## Architecture

### Game Loop
1. **Action Execution**: Agent action applied to Player 1
2. **Opponent Logic**: Player 2 moves (stationary/random/trained)
3. **Physics Update**: Ball movement, collisions, friction
4. **Reward Calculation**: Sparse + shaped rewards
5. **Termination Check**: Goal scored or time limit reached

### Training Pipeline
1. **Ball Control**: Learn to approach and touch ball
2. **Shooting**: Learn to kick toward goal
3. **Full Game**: Combine skills against active opponent

## Development Phases

- **Phase 1**: Game engine with physics and rendering ✓
- **Phase 2**: Gymnasium environment wrapper ✓
- **Phase 3**: Single agent PPO training ✓
- **Phase 4**: Two-agent self-play (planned)
- **Phase 5**: Advanced reward shaping ✓
- **Phase 6**: Visualization and evaluation tools ✓

## License

MIT License - Feel free to use and modify for your own projects.

## Acknowledgments

- Built with [pygame](https://www.pygame.org/)
- RL training with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Environment API follows [Gymnasium](https://gymnasium.farama.org/) standard
