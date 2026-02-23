# 2D Football Game Engine

A complete 2D top-down football game engine built with pygame, featuring physics-driven ball mechanics, rectangular players with rotation-based movement, and goal scoring.

## Features

- Physics-driven ball with friction and bouncing
- Rotatable rectangular players with direction-based movement
- Kick arc mechanics for realistic ball kicking
- Dribbling system when moving with the ball
- Goal detection and scoring
- Wall collision detection
- Time-limited matches

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Project Structure

```
football_game/
├── main.py              # Entry point with game loop
├── config.py            # Constants and configuration
├── state.py             # GameState dataclass
├── renderer.py          # Pygame rendering functions
├── entities.py          # Player and Ball classes
├── physics.py           # Collision detection and physics
└── tests/               # Test suite
```
