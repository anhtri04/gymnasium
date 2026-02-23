import math
from dataclasses import dataclass
from typing import Tuple
from config import PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_SPEED, PLAYER_ROTATION_SPEED

@dataclass
class Player:
    x: float
    y: float
    angle: float  # in degrees, 0 = facing right
    color: Tuple[int, int, int]
    
    def rotate(self, delta_angle: float):
        """Rotate player by delta_angle degrees."""
        self.angle = (self.angle + delta_angle) % 360
    
    def get_facing_vector(self) -> Tuple[float, float]:
        """Return (dx, dy) unit vector of facing direction."""
        rad = math.radians(self.angle)
        return (math.cos(rad), math.sin(rad))
    
    def move_forward(self, distance: float = PLAYER_SPEED):
        """Move forward in facing direction."""
        dx, dy = self.get_facing_vector()
        self.x += dx * distance
        self.y += dy * distance
    
    def move_backward(self, distance: float = PLAYER_SPEED):
        """Move backward (reverse of facing direction)."""
        dx, dy = self.get_facing_vector()
        self.x -= dx * distance
        self.y -= dy * distance
    
    def get_rect_corners(self) -> list:
        """Return the four corners of the rotated rectangle."""
        half_w = PLAYER_WIDTH / 2
        half_h = PLAYER_HEIGHT / 2
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # Corners relative to center
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]
        
        # Rotate and translate
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + self.x
            ry = cx * sin_a + cy * cos_a + self.y
            rotated.append((rx, ry))
        
        return rotated

@dataclass
class Ball:
    x: float
    y: float
    vx: float = 0
    vy: float = 0
    
    def update(self):
        """Update ball position with velocity and apply friction."""
        from config import BALL_FRICTION
        
        # Move
        self.x += self.vx
        self.y += self.vy
        
        # Apply friction
        self.vx *= BALL_FRICTION
        self.vy *= BALL_FRICTION
        
        # Stop if very slow
        if abs(self.vx) < 0.01:
            self.vx = 0
        if abs(self.vy) < 0.01:
            self.vy = 0
