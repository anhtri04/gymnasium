import math
from config import (
    FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT,
    BALL_RADIUS, BALL_BOUNCE_DAMPING, WALL_BOUNCE_DAMPING,
    GOAL_WIDTH, PLAYER_WIDTH, PLAYER_HEIGHT
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
