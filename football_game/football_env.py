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
    
    def __init__(self, render_mode: Optional[str] = None, opponent_type: str = "stationary"):
        super().__init__()
        
        self.render_mode = render_mode
        self.opponent_type = opponent_type
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.state.reset_positions()
        info = {'episode_time': 0.0, 'score1': 0, 'score2': 0}
        return self._get_obs(), info
    
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
        
        # Move opponent (player 2) based on opponent_type
        if self.opponent_type == "random":
            # Random action for player 2
            opponent_action = self.action_space.sample()
            if opponent_action == 0:
                self.state.player2.move_forward(PLAYER_SPEED)
            elif opponent_action == 1:
                self.state.player2.move_backward(PLAYER_SPEED)
            elif opponent_action == 2:
                self.state.player2.rotate(PLAYER_ROTATION_SPEED)
            elif opponent_action == 3:
                self.state.player2.rotate(-PLAYER_ROTATION_SPEED)
            elif opponent_action == 4:
                if is_ball_in_kick_arc(self.state.player2, self.state.ball):
                    kick_ball(self.state.player2, self.state.ball)
        elif self.opponent_type == "stationary":
            # Do nothing
            pass
        
        # Update physics
        self.state.ball.update()
        check_ball_wall_collision(self.state.ball)
        check_player_wall_collision(self.state.player1)
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
    
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
