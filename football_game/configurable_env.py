"""
Configurable environment wrapper for curriculum learning.
Reads YAML config and sets up appropriate training environment.
"""
import yaml
import math
import numpy as np
from typing import Dict, Any, Optional

from football_env import FootballEnv
from entities import Ball


class ConfigurableFootballEnv(FootballEnv):
    """Football environment with config-based setup."""
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None, render_mode: Optional[str] = None):
        """
        Initialize with config.
        
        Args:
            config: Configuration dict
            config_path: Path to YAML config file
            render_mode: Render mode ('human', 'rgb_array', or None)
        """
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.train_config = yaml.safe_load(f)
        elif config:
            self.train_config = config
        else:
            self.train_config = self._default_config()
        
        self.mode = self.train_config.get('mode', 'full_game')
        
        # Initialize parent with no opponent for training modes
        opponent_type = None if self.mode in ['ball_control', 'shooting'] else 'stationary'
        super().__init__(render_mode=render_mode, opponent_type=opponent_type)
        
        # Apply config
        self._apply_config()
        
        # Override observation space based on mode
        from gymnasium import spaces
        if self.mode == 'ball_control':
            # 3 (player) + 4 * num_balls (balls)
            num_balls = len(self.balls) if hasattr(self, 'balls') else 1
            obs_dim = 3 + num_balls * 4
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
        elif self.mode == 'shooting':
            # 9 values: player(3) + ball(4) + distance(1) + angle_diff(1)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(9,), dtype=np.float32
            )
    
    def _default_config(self) -> Dict:
        """Default config for full game."""
        from config import FIELD_WIDTH, FIELD_HEIGHT, FIELD_X, FIELD_Y
        return {
            'mode': 'full_game',
            'field': {
                'width': FIELD_WIDTH,
                'height': FIELD_HEIGHT,
                'start_x': FIELD_X,
                'start_y': FIELD_Y
            },
            'rewards': {
                'touch': 0.01,
                'kick': 0.05,
                'progress': 0.1,
                'goal': 1.0
            }
        }
    
    def _apply_config(self):
        """Apply configuration to environment."""
        if self.mode == 'ball_control':
            self._setup_ball_control()
        elif self.mode == 'shooting':
            self._setup_shooting()
    
    def _setup_ball_control(self):
        """Setup ball control mode."""
        field_cfg = self.train_config['field']
        balls_cfg = self.train_config['balls']
        agent_cfg = self.train_config['agent']
        
        # Set field bounds
        self.field_start_x = field_cfg['start_x']
        self.field_start_y = field_cfg['start_y']
        self.field_width = field_cfg['width']
        self.field_height = field_cfg['height']
        
        # Create multiple balls
        self.balls = []
        for ball_cfg in balls_cfg:
            ball = Ball(
                x=self.field_start_x + ball_cfg['x'],
                y=self.field_start_y + ball_cfg['y'],
                vx=ball_cfg.get('vx', 0),
                vy=ball_cfg.get('vy', 0)
            )
            self.balls.append(ball)
        
        # Set agent position
        self.state.player1.x = agent_cfg['x']
        self.state.player1.y = agent_cfg['y']
        self.state.player1.angle = agent_cfg['angle']
        
        # Use first ball as main ball
        if self.balls:
            self.state.ball = self.balls[0]
    
    def _setup_shooting(self):
        """Setup shooting mode."""
        field_cfg = self.train_config['field']
        agent_cfg = self.train_config['agent']
        balls_cfg = self.train_config['balls']
        
        # Set field bounds
        self.field_start_x = field_cfg['start_x']
        self.field_start_y = field_cfg['start_y']
        self.field_width = field_cfg['width']
        self.field_height = field_cfg['height']
        
        # Set goal width
        goal_cfg = self.train_config.get('goal', {})
        self.goal_width = goal_cfg.get('width', 100)
        
        # Set agent position
        self.state.player1.x = agent_cfg['x']
        self.state.player1.y = agent_cfg['y']
        self.state.player1.angle = agent_cfg['angle']
        
        # Set ball position
        ball_cfg = balls_cfg[0] if balls_cfg else {}
        if ball_cfg.get('at_feet'):
            # Place at feet
            offset = 30
            rad = math.radians(self.state.player1.angle)
            self.state.ball.x = self.state.player1.x + math.cos(rad) * offset
            self.state.ball.y = self.state.player1.y + math.sin(rad) * offset
        else:
            self.state.ball.x = self.field_start_x + ball_cfg.get('x', 0)
            self.state.ball.y = self.field_start_y + ball_cfg.get('y', 0)
        
        self.state.ball.vx = 0
        self.state.ball.vy = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Re-apply config after parent reset
        self._apply_config()
        
        # Update info with config
        info['mode'] = self.mode
        info['stage'] = self.train_config.get('stage', 'default')
        
        return obs, info
    
    def _get_obs(self) -> np.ndarray:
        """Get observation based on mode."""
        if self.mode == 'ball_control':
            return self._get_ball_control_obs()
        elif self.mode == 'shooting':
            return self._get_shooting_obs()
        else:
            return super()._get_obs()
    
    def _get_ball_control_obs(self) -> np.ndarray:
        """Observation for ball control."""
        from config import SCREEN_WIDTH, SCREEN_HEIGHT
        
        obs = [
            self.state.player1.x / SCREEN_WIDTH,
            self.state.player1.y / SCREEN_HEIGHT,
            self.state.player1.angle / 360.0
        ]
        
        # Add all balls
        for ball in self.balls:
            obs.extend([
                ball.x / SCREEN_WIDTH,
                ball.y / SCREEN_HEIGHT,
                (ball.vx / 20.0 + 1) / 2,
                (ball.vy / 20.0 + 1) / 2
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_shooting_obs(self) -> np.ndarray:
        """Observation for shooting."""
        from config import SCREEN_WIDTH, SCREEN_HEIGHT, FIELD_WIDTH, FIELD_HEIGHT, FIELD_X, FIELD_Y
        
        p1_x = self.state.player1.x / SCREEN_WIDTH
        p1_y = self.state.player1.y / SCREEN_HEIGHT
        p1_angle = self.state.player1.angle / 360.0
        
        ball_x = self.state.ball.x / SCREEN_WIDTH
        ball_y = self.state.ball.y / SCREEN_HEIGHT
        ball_vx = (self.state.ball.vx / 20.0 + 1) / 2
        ball_vy = (self.state.ball.vy / 20.0 + 1) / 2
        
        # Distance and angle to goal
        goal_x = self.field_start_x + self.field_width
        goal_y = self.field_start_y + self.field_height / 2
        dx = goal_x - self.state.player1.x
        dy = goal_y - self.state.player1.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        angle_to_goal = math.degrees(math.atan2(dy, dx))
        angle_diff = (angle_to_goal - self.state.player1.angle + 180) % 360 - 180
        
        obs = np.array([
            p1_x, p1_y, p1_angle,
            ball_x, ball_y, ball_vx, ball_vy,
            min(dist / FIELD_WIDTH, 1.0),
            (angle_diff + 180) / 360
        ], dtype=np.float32)
        
        return np.clip(obs, 0.0, 1.0)
