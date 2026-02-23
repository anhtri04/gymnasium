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
