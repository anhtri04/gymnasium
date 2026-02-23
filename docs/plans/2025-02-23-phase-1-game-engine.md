# Phase 1 — Game Engine Implementation Plan
> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
**Goal:** Build a complete 2D top-down football game engine in pygame with physics, rendering, and keyboard controls — no AI yet.
**Architecture:** Pure pygame implementation with physics-driven ball, rectangular players with rotation-based movement, kick arc mechanics, and wall collision. All state managed in a single `GameState` class. Rendering separated from physics update for future Gym integration.
**Tech Stack:** Python, pygame, dataclasses for state, vector math for physics
---
## Project Structure
```
football_game/
├── main.py              # Entry point with game loop
├── config.py            # Constants (screen size, colors, physics params)
├── state.py             # GameState dataclass and physics
├── renderer.py          # Pygame rendering functions
├── entities.py          # Player and Ball classes
├── physics.py           # Collision detection and physics helpers
├── controls.py          # Keyboard input handling
└── tests/
    ├── test_state.py
    ├── test_physics.py
    └── test_entities.py
```
---
## Pre-Phase: Setup
### Task 0: Initialize Project and Dependencies
**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `README.md` (basic)
**Step 1: Create requirements.txt**
```
pygame>=2.5.0
```
**Step 2: Install dependencies**
Run: `pip install -r requirements.txt`
Expected: Successfully installed pygame
**Step 3: Create config.py with constants**
```python
import pygame
# Screen
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
# Colors
FIELD_GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
# Field dimensions (in pixels)
FIELD_WIDTH = 1000
FIELD_HEIGHT = 600
FIELD_X = (SCREEN_WIDTH - FIELD_WIDTH) // 2
FIELD_Y = (SCREEN_HEIGHT - FIELD_HEIGHT) // 2
# Goals
GOAL_WIDTH = 100  # Opening size
GOAL_DEPTH = 20
# Players
PLAYER_WIDTH = 30
PLAYER_HEIGHT = 20
PLAYER_SPEED = 5
PLAYER_ROTATION_SPEED = 5  # degrees per frame
PLAYER_KICK_DISTANCE = 60
PLAYER_KICK_ANGLE = 45  # degrees for kick arc (half-angle)
# Ball
BALL_RADIUS = 10
BALL_FRICTION = 0.98  # velocity multiplier per frame
BALL_BOUNCE_DAMPING = 0.7
BALL_MAX_SPEED = 20
# Physics
WALL_BOUNCE_DAMPING = 0.8
EPISODE_TIME_LIMIT = 60  # seconds
```
**Step 4: Commit**
```bash
git add requirements.txt config.py README.md
git commit -m "feat: add project structure and configuration"
```
---
## Part A: Field Rendering
### Task 1: Create Game Window and Field
**Files:**
- Create: `main.py`
- Create: `renderer.py`
**Step 1: Write test**
```python
# tests/test_renderer.py
def test_field_rect_position():
    from config import FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT
    assert FIELD_X == 100
    assert FIELD_Y == 100
    assert FIELD_WIDTH == 1000
    assert FIELD_HEIGHT == 600
```
**Step 2: Run test**
Run: `pytest tests/test_renderer.py::test_field_rect_position -v`
Expected: PASS
**Step 3: Implement main.py and renderer.py**
main.py:
```python
import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2D Football Game")
        self.clock = pygame.time.Clock()
        self.running = True
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def update(self):
        pass
    
    def render(self):
        from renderer import render_field
        render_field(self.screen)
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()
if __name__ == "__main__":
    game = Game()
    game.run()
```
renderer.py:
```python
import pygame
from config import (
    FIELD_GREEN, WHITE, SCREEN_WIDTH, SCREEN_HEIGHT,
    FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT,
    GOAL_WIDTH, GOAL_DEPTH
)
def render_field(screen):
    # Clear screen
    screen.fill((50, 50, 50))
    
    # Draw field
    pygame.draw.rect(screen, FIELD_GREEN, (FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT))
    
    # Draw border
    pygame.draw.rect(screen, WHITE, (FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT), 2)
    
    # Draw center line
    center_x = FIELD_X + FIELD_WIDTH // 2
    pygame.draw.line(screen, WHITE, (center_x, FIELD_Y), (center_x, FIELD_Y + FIELD_HEIGHT), 2)
    
    # Draw center circle
    center_y = FIELD_Y + FIELD_HEIGHT // 2
    pygame.draw.circle(screen, WHITE, (center_x, center_y), 60, 2)
    
    # Draw center spot
    pygame.draw.circle(screen, WHITE, (center_x, center_y), 4)
    
    # Draw goals (openings on left and right)
    goal_top = FIELD_Y + (FIELD_HEIGHT - GOAL_WIDTH) // 2
    
    # Left goal
    pygame.draw.rect(screen, WHITE, (FIELD_X - GOAL_DEPTH, goal_top, GOAL_DEPTH, GOAL_WIDTH), 2)
    
    # Right goal
    pygame.draw.rect(screen, WHITE, (FIELD_X + FIELD_WIDTH, goal_top, GOAL_DEPTH, GOAL_WIDTH), 2)
```
**Step 4: Test rendering**
Run: `python main.py`
Expected: Window opens showing green field with white lines, center circle, and goals. Press ESC to close.
**Step 5: Commit**
```bash
git add main.py renderer.py tests/test_renderer.py
git commit -m "feat: add field rendering with goals and center circle"
```

---
## Part B: Player Entity
### Task 2: Create Player Class with Position and Facing
**Files:**
- Create: `entities.py`
**Step 1: Write test**
```python
# tests/test_entities.py
import math
from entities import Player
def test_player_creation():
    player = Player(x=500, y=400, angle=0, color=(255, 0, 0))
    assert player.x == 500
    assert player.y == 400
    assert player.angle == 0
    assert player.color == (255, 0, 0)
def test_player_rotation():
    player = Player(x=0, y=0, angle=0, color=(255, 0, 0))
    player.rotate(90)
    assert player.angle == 90
    player.rotate(370)  # Should wrap to 10
    assert player.angle == 10
```
**Step 2: Run test (should fail)**
Run: `pytest tests/test_entities.py::test_player_creation -v`
Expected: FAIL - Player not defined
**Step 3: Implement entities.py**
```python
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
```
**Step 4: Run test (should pass)**
Run: `pytest tests/test_entities.py -v`
Expected: PASS
**Step 5: Commit**
```bash
git add entities.py tests/test_entities.py
git commit -m "feat: add Player entity with rotation and movement"
```
---
### Task 3: Render Player with Rotation
**Files:**
- Modify: `renderer.py`
- Modify: `main.py`
**Step 1: Write test**
```python
# tests/test_renderer.py
def test_player_render_no_crash():
    import pygame
    from renderer import render_player
    from entities import Player
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    player = Player(x=400, y=300, angle=45, color=(255, 0, 0))
    
    try:
        render_player(screen, player)
        assert True
    except Exception as e:
        assert False, f"render_player raised {e}"
    
    pygame.quit()
```
**Step 2: Run test (should fail)**
Run: `pytest tests/test_renderer.py::test_player_render_no_crash -v`
Expected: FAIL - render_player not defined
**Step 3: Implement rendering**
Add to renderer.py:
```python
def render_player(screen, player):
    """Render player as rotated rectangle."""
    from config import PLAYER_WIDTH, PLAYER_HEIGHT
    
    # Get corners
    corners = player.get_rect_corners()
    
    # Draw filled rectangle
    pygame.draw.polygon(screen, player.color, corners)
    
    # Draw border
    pygame.draw.polygon(screen, WHITE, corners, 2)
    
    # Draw direction indicator (small line from center)
    dx, dy = player.get_facing_vector()
    nose_x = player.x + dx * PLAYER_WIDTH * 0.6
    nose_y = player.y + dy * PLAYER_WIDTH * 0.6
    pygame.draw.line(screen, WHITE, (player.x, player.y), (nose_x, nose_y), 2)
```
Update main.py:
```python
import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from entities import Player
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2D Football Game")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Create players
        self.player1 = Player(
            x=SCREEN_WIDTH // 2 - 200,
            y=SCREEN_HEIGHT // 2,
            angle=0,
            color=RED
        )
        self.player2 = Player(
            x=SCREEN_WIDTH // 2 + 200,
            y=SCREEN_HEIGHT // 2,
            angle=180,
            color=BLUE
        )
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def update(self):
        pass
    
    def render(self):
        from renderer import render_field, render_player
        render_field(self.screen)
        render_player(self.screen, self.player1)
        render_player(self.screen, self.player2)
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()
if __name__ == "__main__":
    from config import RED, BLUE
    game = Game()
    game.run()
```
**Step 4: Run test**
Run: `pytest tests/test_renderer.py::test_player_render_no_crash -v`
Expected: PASS
**Step 5: Visual test**
Run: `python main.py`
Expected: Two rectangles visible (red on left, blue on right), both with direction lines. ESC to close.
**Step 6: Commit**
```bash
git add renderer.py main.py
git commit -m "feat: render players with rotation and direction indicator"
```
---
## Part C: Ball Entity
### Task 4: Create Ball Class with Physics
**Files:**
- Modify: `entities.py`
**Step 1: Write test**
```python
# tests/test_entities.py
def test_ball_creation():
    from entities import Ball
    ball = Ball(x=500, y=400, vx=5, vy=3)
    assert ball.x == 500
    assert ball.y == 400
    assert ball.vx == 5
    assert ball.vy == 3
def test_ball_update_with_friction():
    from entities import Ball
    from config import BALL_FRICTION
    ball = Ball(x=100, y=100, vx=10, vy=0)
    ball.update()
    assert ball.vx == 10 * BALL_FRICTION
    assert ball.x == 100 + 10 * BALL_FRICTION
```
**Step 2: Run test (should fail)**
Run: `pytest tests/test_entities.py::test_ball_creation -v`
Expected: FAIL - Ball not defined
**Step 3: Implement Ball in entities.py**
```python
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
```
**Step 4: Run test (should pass)**
Run: `pytest tests/test_entities.py::test_ball_creation tests/test_entities.py::test_ball_update_with_friction -v`
Expected: PASS
**Step 5: Commit**
```bash
git add entities.py tests/test_entities.py
git commit -m "feat: add Ball entity with velocity and friction"
```

---

### Task 5: Render Ball and Add to Game

**Files:**
- Modify: `renderer.py`
- Modify: `main.py`

**Step 1: Write test**

```python
# tests/test_renderer.py
def test_ball_render_no_crash():
    import pygame
    from renderer import render_ball
    from entities import Ball
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    ball = Ball(x=400, y=300, vx=5, vy=5)
    
    try:
        render_ball(screen, ball)
        assert True
    except Exception as e:
        assert False, f"render_ball raised {e}"
    
    pygame.quit()
```

**Step 2: Run test (should fail)**

Run: `pytest tests/test_renderer.py::test_ball_render_no_crash -v`
Expected: FAIL - render_ball not defined

**Step 3: Implement rendering**

Add to renderer.py:
```python
def render_ball(screen, ball):
    """Render ball as circle."""
    from config import BALL_RADIUS, WHITE
    import pygame
    
    # Draw ball
    pygame.draw.circle(screen, WHITE, (int(ball.x), int(ball.y)), BALL_RADIUS)
    
    # Draw velocity indicator (optional - for debugging)
    end_x = int(ball.x + ball.vx * 3)
    end_y = int(ball.y + ball.vy * 3)
    pygame.draw.line(screen, (255, 255, 0), (int(ball.x), int(ball.y)), (end_x, end_y), 2)
```

Update main.py:
```python
# In __init__
self.ball = Ball(x=SCREEN_WIDTH // 2, y=SCREEN_HEIGHT // 2, vx=0, vy=0)

# In update
self.ball.update()

# In render
from renderer import render_field, render_player, render_ball
render_field(self.screen)
render_ball(self.screen, self.ball)
render_player(self.screen, self.player1)
render_player(self.screen, self.player2)
```

**Step 4: Run test (should pass)**

Run: `pytest tests/test_renderer.py::test_ball_render_no_crash -v`
Expected: PASS

**Step 5: Visual test**

Run: `python main.py`
Expected: White ball in center with yellow velocity indicator. ESC to close.

**Step 6: Commit**

```bash
git add renderer.py main.py
git commit -m "feat: render ball with velocity indicator"
```

---

## Part D: Physics

### Task 6: Implement Wall Collision and Bouncing

**Files:**
- Create: `physics.py`
- Modify: `entities.py` (add collision method to Ball)

**Step 1: Write test**

```python
# tests/test_physics.py
def test_ball_wall_bounce():
    from physics import check_ball_wall_collision
    from entities import Ball
    from config import FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT
    
    # Ball hitting right wall
    ball = Ball(x=FIELD_X + FIELD_WIDTH + 5, y=FIELD_Y + 100, vx=5, vy=0)
    check_ball_wall_collision(ball)
    assert ball.vx < 0  # Should bounce back
    assert ball.x <= FIELD_X + FIELD_WIDTH  # Should be inside field
```

**Step 2: Run test (should fail)**

Run: `pytest tests/test_physics.py::test_ball_wall_bounce -v`
Expected: FAIL - check_ball_wall_collision not defined

**Step 3: Implement physics.py**

```python
import math
from config import (
    FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT,
    BALL_RADIUS, BALL_BOUNCE_DAMPING, WALL_BOUNCE_DAMPING,
    GOAL_WIDTH
)

def check_ball_wall_collision(ball):
    """Check and handle ball collision with field boundaries."""
    left = FIELD_X
    right = FIELD_X + FIELD_WIDTH
    top = FIELD_Y
    bottom = FIELD_Y + FIELD_HEIGHT
    
    # Calculate goal y-range
    goal_top = FIELD_Y + (FIELD_HEIGHT - GOAL_WIDTH) // 2
    goal_bottom = goal_top + GOAL_WIDTH
    
    # Left wall (check if in goal or not)
    if ball.x - BALL_RADIUS < left:
        # Check if in goal opening
        if goal_top < ball.y < goal_bottom:
            # Ball is in goal - don't bounce
            pass
        else:
            # Bounce off wall
            ball.x = left + BALL_RADIUS
            ball.vx = abs(ball.vx) * WALL_BOUNCE_DAMPING
    
    # Right wall
    if ball.x + BALL_RADIUS > right:
        # Check if in goal opening
        if goal_top < ball.y < goal_bottom:
            # Ball is in goal - don't bounce
            pass
        else:
            # Bounce off wall
            ball.x = right - BALL_RADIUS
            ball.vx = -abs(ball.vx) * WALL_BOUNCE_DAMPING
    
    # Top wall
    if ball.y - BALL_RADIUS < top:
        ball.y = top + BALL_RADIUS
        ball.vy = abs(ball.vy) * WALL_BOUNCE_DAMPING
    
    # Bottom wall
    if ball.y + BALL_RADIUS > bottom:
        ball.y = bottom - BALL_RADIUS
        ball.vy = -abs(ball.vy) * WALL_BOUNCE_DAMPING

def check_player_wall_collision(player):
    """Keep player inside field boundaries."""
    from entities import Player
    from config import PLAYER_WIDTH, PLAYER_HEIGHT
    
    # Get bounding box of rotated player
    corners = player.get_rect_corners()
    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)
    
    # Push back if out of bounds
    if min_x < FIELD_X:
        player.x += FIELD_X - min_x
    if max_x > FIELD_X + FIELD_WIDTH:
        player.x -= max_x - (FIELD_X + FIELD_WIDTH)
    if min_y < FIELD_Y:
        player.y += FIELD_Y - min_y
    if max_y > FIELD_Y + FIELD_HEIGHT:
        player.y -= max_y - (FIELD_Y + FIELD_HEIGHT)
```

**Step 4: Run test (should pass)**

Run: `pytest tests/test_physics.py::test_ball_wall_bounce -v`
Expected: PASS

**Step 5: Integrate into game loop**

Update main.py:
```python
from physics import check_ball_wall_collision, check_player_wall_collision

# In update method:
def update(self):
    from physics import check_ball_wall_collision, check_player_wall_collision
    
    # Update ball
    self.ball.update()
    check_ball_wall_collision(self.ball)
    
    # Keep players in bounds
    check_player_wall_collision(self.player1)
    check_player_wall_collision(self.player2)
```

**Step 6: Visual test**

Run: `python main.py`
Expected: Ball stays on field, bounces off walls (except goal openings). ESC to close.

**Step 7: Commit**

```bash
git add physics.py tests/test_physics.py main.py
git commit -m "feat: add wall collision detection and bouncing"
```

---

## Part E: Kick Mechanics

### Task 7: Implement Kick Arc Detection

**Files:**
- Modify: `physics.py`

**Step 1: Write test**

```python
# tests/test_physics.py
import math

def test_ball_in_kick_arc():
    from physics import is_ball_in_kick_arc
    from entities import Player, Ball
    
    # Player facing right (0 degrees), ball directly in front
    player = Player(x=500, y=400, angle=0, color=(255, 0, 0))
    ball = Ball(x=550, y=400)  # 50 pixels in front
    
    assert is_ball_in_kick_arc(player, ball) == True
    
    # Ball behind player
    ball2 = Ball(x=450, y=400)
    assert is_ball_in_kick_arc(player, ball2) == False
    
    # Ball at 60 degrees (outside 45 degree arc)
    ball3 = Ball(x=500, y=500)  # 100 pixels down (60 degrees)
    assert is_ball_in_kick_arc(player, ball3) == False
```

**Step 2: Run test (should fail)**

Run: `pytest tests/test_physics.py::test_ball_in_kick_arc -v`
Expected: FAIL - is_ball_in_kick_arc not defined

**Step 3: Implement kick arc logic**

Add to physics.py:
```python
def is_ball_in_kick_arc(player, ball):
    """Check if ball is within player's kick arc."""
    from config import PLAYER_KICK_DISTANCE, PLAYER_KICK_ANGLE
    
    # Calculate vector from player to ball
    dx = ball.x - player.x
    dy = ball.y - player.y
    distance = math.sqrt(dx * dx + dy * dy)
    
    # Check distance
    if distance > PLAYER_KICK_DISTANCE:
        return False
    
    # Check angle
    ball_angle = math.degrees(math.atan2(dy, dx))
    angle_diff = abs((ball_angle - player.angle + 180) % 360 - 180)
    
    return angle_diff <= PLAYER_KICK_ANGLE
```

**Step 4: Run test (should pass)**

Run: `pytest tests/test_physics.py::test_ball_in_kick_arc -v`
Expected: PASS

**Step 5: Commit**

```bash
git add physics.py tests/test_physics.py
git commit -m "feat: add kick arc collision detection"
```

---
### Task 8: Implement Kick Action
**Files:**
- Modify: `physics.py`
**Step 1: Write test**
```python
# tests/test_physics.py
def test_kick_ball():
    from physics import kick_ball, is_ball_in_kick_arc
    from entities import Player, Ball
    
    player = Player(x=500, y=400, angle=0, color=(255, 0, 0))
    ball = Ball(x=550, y=400, vx=0, vy=0)
    
    # Kick the ball
    kick_ball(player, ball)
    
    # Ball should have velocity in facing direction
    assert ball.vx > 0
    assert ball.vy == 0
```
**Step 2: Run test (should fail)**
Run: `pytest tests/test_physics.py::test_kick_ball -v`
Expected: FAIL - kick_ball not defined
**Step 3: Implement kick**
Add to physics.py:
```python
def kick_ball(player, ball):
    """Apply kick force to ball in player's facing direction."""
    from config import BALL_MAX_SPEED
    
    dx, dy = player.get_facing_vector()
    kick_power = 15  # Base kick power
    
    # Set ball velocity
    ball.vx = dx * kick_power
    ball.vy = dy * kick_power
    
    # Clamp to max speed
    speed = math.sqrt(ball.vx * ball.vx + ball.vy * ball.vy)
    if speed > BALL_MAX_SPEED:
        ball.vx = (ball.vx / speed) * BALL_MAX_SPEED
        ball.vy = (ball.vy / speed) * BALL_MAX_SPEED
```
**Step 4: Run test (should pass)**
Run: `pytest tests/test_physics.py::test_kick_ball -v`
Expected: PASS
**Step 5: Commit**
```bash
git add physics.py tests/test_physics.py
git commit -m "feat: add ball kick mechanic"
```
---
### Task 9: Implement Dribbling
**Files:**
- Modify: `physics.py`
**Step 1: Write test**
```python
# tests/test_physics.py
def test_dribble_ball():
    from physics import dribble_ball
    from entities import Player, Ball
    
    player = Player(x=500, y=400, angle=0, color=(255, 0, 0))
    ball = Ball(x=540, y=400, vx=0, vy=0)  # Close and aligned
    
    # Move player forward
    player.move_forward(5)
    
    # Dribble should move ball with player
    dribble_ball(player, ball)
    
    # Ball should be in front of player now
    assert ball.x > player.x
```
**Step 2: Run test (should fail)**
Run: `pytest tests/test_physics.py::test_dribble_ball -v`
Expected: FAIL - dribble_ball not defined
**Step 3: Implement dribbling**
Add to physics.py:
```python
def dribble_ball(player, ball):
    """Move ball with player when close and aligned (dribbling)."""
    from config import PLAYER_KICK_DISTANCE
    
    # Calculate distance to ball
    dx = ball.x - player.x
    dy = ball.y - player.y
    distance = math.sqrt(dx * dx + dy * dy)
    
    # Only dribble if ball is close
    if distance > PLAYER_KICK_DISTANCE * 0.6:
        return
    
    # Calculate ball angle relative to player
    ball_angle = math.degrees(math.atan2(dy, dx))
    angle_diff = abs((ball_angle - player.angle + 180) % 360 - 180)
    
    # Only dribble if ball is in front (within 60 degrees)
    if angle_diff > 60:
        return
    
    # Move ball to stay in front of player
    from config import PLAYER_WIDTH
    offset_distance = PLAYER_WIDTH * 0.8
    dx, dy = player.get_facing_vector()
    ball.x = player.x + dx * offset_distance
    ball.y = player.y + dy * offset_distance
    
    # Give ball slight forward velocity
    ball.vx = dx * 2
    ball.vy = dy * 2
```
**Step 4: Run test (should pass)**
Run: `pytest tests/test_physics.py::test_dribble_ball -v`
Expected: PASS
**Step 5: Commit**
```bash
git add physics.py tests/test_physics.py
git commit -m "feat: add dribbling mechanic"
```
---
## Part F: Goal Detection and Scoring
### Task 10: Implement Goal Detection
**Files:**
- Modify: `physics.py`
**Step 1: Write test**
```python
# tests/test_physics.py
def test_goal_detection():
    from physics import check_goal
    from entities import Ball
    from config import FIELD_X, FIELD_Y, FIELD_HEIGHT, GOAL_WIDTH
    
    # Ball in left goal
    goal_top = FIELD_Y + (FIELD_HEIGHT - GOAL_WIDTH) // 2
    ball = Ball(x=FIELD_X - 10, y=goal_top + 10, vx=-5, vy=0)
    result = check_goal(ball)
    assert result == "left"
    
    # Ball in right goal
    from config import FIELD_WIDTH
    ball2 = Ball(x=FIELD_X + FIELD_WIDTH + 10, y=goal_top + 10, vx=5, vy=0)
    result2 = check_goal(ball2)
    assert result2 == "right"
    
    # Ball not in goal
    ball3 = Ball(x=FIELD_X + 100, y=FIELD_Y + 100)
    result3 = check_goal(ball3)
    assert result3 is None
```
**Step 2: Run test (should fail)**
Run: `pytest tests/test_physics.py::test_goal_detection -v`
Expected: FAIL - check_goal not defined
**Step 3: Implement goal detection**
Add to physics.py:
```python
def check_goal(ball):
    """Check if ball is in either goal. Returns 'left', 'right', or None."""
    from config import FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT, GOAL_WIDTH, BALL_RADIUS
    
    # Calculate goal y-range
    goal_top = FIELD_Y + (FIELD_HEIGHT - GOAL_WIDTH) // 2
    goal_bottom = goal_top + GOAL_WIDTH
    
    # Check left goal
    if ball.x + BALL_RADIUS < FIELD_X and goal_top < ball.y < goal_bottom:
        return "left"
    
    # Check right goal
    if ball.x - BALL_RADIUS > FIELD_X + FIELD_WIDTH and goal_top < ball.y < goal_bottom:
        return "right"
    
    return None
```
**Step 4: Run test (should pass)**
Run: `pytest tests/test_physics.py::test_goal_detection -v`
Expected: PASS
**Step 5: Commit**
```bash
git add physics.py tests/test_physics.py
git commit -m "feat: add goal detection system"
```
---
### Task 11: Implement Game Reset After Goal
**Files:**
- Create: `state.py`
**Step 1: Write test**
```python
# tests/test_state.py
def test_game_reset():
    from state import GameState
    from entities import Player, Ball
    
    state = GameState()
    
    # Move everything
    state.player1.x = 100
    state.player2.x = 900
    state.ball.x = 50  # Left goal
    
    # Reset
    state.reset_positions()
    
    # Check positions reset
    assert state.player1.x == state._start_p1_x
    assert state.player2.x == state._start_p2_x
    assert state.ball.x == state._start_ball_x
    assert state.ball.vx == 0
```
**Step 2: Run test (should fail)**
Run: `pytest tests/test_state.py::test_game_reset -v`
Expected: FAIL - GameState not defined
**Step 3: Implement GameState**
Create `state.py`:
```python
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
    
    def update_episode_time(self, dt: float):
        """Update episode timer."""
        from config import EPISODE_TIME_LIMIT
        self.episode_time += dt
        return self.episode_time >= EPISODE_TIME_LIMIT
```
**Step 4: Run test (should pass)**
Run: `pytest tests/test_state.py::test_game_reset -v`
Expected: PASS
**Step 5: Commit**
```bash
git add state.py tests/test_state.py
git commit -m "feat: add GameState with reset functionality"
```
---
### Task 12: Add Time Limit and Scoreboard
**Files:**
- Modify: `state.py`
- Modify: `renderer.py`
- Modify: `main.py`
**Step 1: Write test**
```python
# tests/test_state.py
def test_episode_timeout():
    from state import GameState
    from config import EPISODE_TIME_LIMIT
    
    state = GameState()
    
    # Advance time
    timed_out = False
    for i in range(70):  # 70 seconds at 1 sec per call
        timed_out = state.update_episode_time(1.0)
    
    assert timed_out == True
```
**Step 2: Run test (should fail)**
Run: `pytest tests/test_state.py::test_episode_timeout -v`
Expected: FAIL - update_episode_time may not return correctly
**Step 3: Update GameState**
Update state.py:
```python
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
```
**Step 4: Run test (should pass)**
Run: `pytest tests/test_state.py::test_episode_timeout -v`
Expected: PASS
**Step 5: Add scoreboard rendering**
Add to renderer.py:
```python
def render_scoreboard(screen, score1, score2):
    """Render scoreboard at top of screen."""
    from config import SCREEN_WIDTH, WHITE, BLACK
    import pygame
    
    # Fonts
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 48)
    small_font = pygame.font.SysFont('Arial', 24)
    
    # Draw score
    score_text = f"{score1} - {score2}"
    text_surface = font.render(score_text, True, WHITE)
    text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, 30))
    screen.blit(text_surface, text_rect)
```
**Step 6: Update main.py with scoreboard and timer**
Update main.py:
```python
import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, EPISODE_TIME_LIMIT
from entities import Player, Ball
from state import GameState
from physics import (
    check_ball_wall_collision,
    check_player_wall_collision,
    check_goal
)
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2D Football Game")
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = GameState()
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def update(self, dt):
        # Update ball
        self.state.ball.update()
        check_ball_wall_collision(self.state.ball)
        
        # Check for goals
        goal = check_goal(self.state.ball)
        if goal == "left":
            self.state.increment_score(2)
            self.state.reset_positions()
        elif goal == "right":
            self.state.increment_score(1)
            self.state.reset_positions()
        
        # Keep players in bounds
        check_player_wall_collision(self.state.player1)
        check_player_wall_collision(self.state.player2)
        
        # Update timer
        timed_out = self.state.update_episode_time(dt)
        if timed_out:
            self.state.reset_positions()
    
    def render(self):
        from renderer import render_field, render_player, render_ball, render_scoreboard
        render_field(self.screen)
        render_ball(self.screen, self.state.ball)
        render_player(self.screen, self.state.player1)
        render_player(self.screen, self.state.player2)
        render_scoreboard(self.screen, self.state.score1, self.state.score2)
        pygame.display.flip()
    
    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.render()
        pygame.quit()
        sys.exit()
if __name__ == "__main__":
    game = Game()
    game.run()
```
**Step 7: Visual test**
Run: `python main.py`
Expected: Scoreboard visible at top, goals increment correctly, reset after goal, timer counts down. ESC to close.
**Step 8: Commit**
```bash
git add state.py renderer.py main.py tests/test_state.py
git commit -m "feat: add scoreboard, time limit, and goal scoring"
```
