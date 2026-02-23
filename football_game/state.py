from dataclasses import dataclass
from typing import Optional
from entities import Player, Ball
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT,
    RED, BLUE
)

@dataclass
class GameState:
    """Complete game state including players, ball, and score."""
    
    # Initialize with default positions
    def __init__(self):
        self.score1 = 0
        self.score2 = 0
        self.episode_time = 0.0
        
        # Starting positions
        self._start_p1_x = FIELD_X + 200
        self._start_p1_y = FIELD_Y + FIELD_HEIGHT // 2
        self._start_p2_x = FIELD_X + FIELD_WIDTH - 200
        self._start_p2_y = FIELD_Y + FIELD_HEIGHT // 2
        self._start_ball_x = FIELD_X + FIELD_WIDTH // 2
        self._start_ball_y = FIELD_Y + FIELD_HEIGHT // 2
        
        # Create entities
        self.player1 = Player(
            x=self._start_p1_x,
            y=self._start_p1_y,
            angle=0,
            color=RED
        )
        self.player2 = Player(
            x=self._start_p2_x,
            y=self._start_p2_y,
            angle=180,
            color=BLUE
        )
        self.ball = Ball(
            x=self._start_ball_x,
            y=self._start_ball_y,
            vx=0,
            vy=0
        )
    
    def reset_positions(self):
        """Reset all entities to starting positions."""
        self.player1.x = self._start_p1_x
        self.player1.y = self._start_p1_y
        self.player1.angle = 0
        
        self.player2.x = self._start_p2_x
        self.player2.y = self._start_p2_y
        self.player2.angle = 180
        
        self.ball.x = self._start_ball_x
        self.ball.y = self._start_ball_y
        self.ball.vx = 0
        self.ball.vy = 0
        
        self.episode_time = 0.0
    
    def update_episode_time(self, dt: float) -> bool:
        """Update episode timer. Returns True if time limit reached."""
        from config import EPISODE_TIME_LIMIT
        self.episode_time += dt
        return self.episode_time >= EPISODE_TIME_LIMIT
    
    def increment_score(self, player: int):
        """Increment score for player 1 or 2."""
        if player == 1:
            self.score1 += 1
        elif player == 2:
            self.score2 += 1
