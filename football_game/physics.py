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

def check_ball_player_collision(ball, player):
    """Check and handle ball collision with a player. Ball bounces off player."""
    # Get player rectangle corners
    corners = player.get_rect_corners()
    
    # Find closest point on rectangle to ball center
    closest_x, closest_y = _closest_point_on_polygon(ball.x, ball.y, corners)
    
    # Calculate distance from ball to closest point
    dx = ball.x - closest_x
    dy = ball.y - closest_y
    distance = math.sqrt(dx * dx + dy * dy)
    
    # Check if collision occurred
    if distance < BALL_RADIUS:
        # Calculate collision normal (direction from closest point to ball)
        if distance > 0:
            nx = dx / distance
            ny = dy / distance
        else:
            # Ball center is exactly on the edge, use player's facing direction perpendicular
            fx, fy = player.get_facing_vector()
            nx = -fy
            ny = fx
        
        # Push ball outside player
        overlap = BALL_RADIUS - distance
        ball.x += nx * overlap
        ball.y += ny * overlap
        
        # Bounce velocity: reflect velocity across normal
        dot_product = ball.vx * nx + ball.vy * ny
        ball.vx = (ball.vx - 2 * dot_product * nx) * WALL_BOUNCE_DAMPING
        ball.vy = (ball.vy - 2 * dot_product * ny) * WALL_BOUNCE_DAMPING
        
        return True
    
    return False

def _closest_point_on_polygon(px, py, corners):
    """Find the closest point on a polygon to point (px, py)."""
    min_dist_sq = float('inf')
    closest_x, closest_y = px, py
    
    # Check each edge of the polygon
    n = len(corners)
    for i in range(n):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n]
        
        # Find closest point on line segment
        cx, cy = _closest_point_on_segment(px, py, x1, y1, x2, y2)
        
        dist_sq = (px - cx) ** 2 + (py - cy) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_x, closest_y = cx, cy
    
    return closest_x, closest_y

def _closest_point_on_segment(px, py, x1, y1, x2, y2):
    """Find closest point on line segment to point (px, py)."""
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return x1, y1
    
    # Project point onto line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    return x1 + t * dx, y1 + t * dy

def check_player_player_collision(player1, player2):
    """Check and handle collision between two players. Players block each other."""
    # Get corners for both players
    corners1 = player1.get_rect_corners()
    corners2 = player2.get_rect_corners()
    
    # Check if any corner of player1 is inside player2
    collision = False
    min_overlap = float('inf')
    push_dx, push_dy = 0, 0
    
    # Check all corners of player1 against player2 edges
    for corner in corners1:
        cx, cy = corner
        # Check if this corner is inside player2 using point-in-polygon
        if _point_in_polygon(cx, cy, corners2):
            # Find closest point on player2 to this corner
            closest_x, closest_y = _closest_point_on_polygon(cx, cy, corners2)
            dx = cx - closest_x
            dy = cy - closest_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_overlap and dist_sq > 0:
                min_overlap = dist_sq
                push_dx = dx
                push_dy = dy
                collision = True
    
    # Check all corners of player2 against player1 edges
    for corner in corners2:
        cx, cy = corner
        # Check if this corner is inside player1
        if _point_in_polygon(cx, cy, corners1):
            # Find closest point on player1 to this corner
            closest_x, closest_y = _closest_point_on_polygon(cx, cy, corners1)
            dx = cx - closest_x
            dy = cy - closest_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_overlap and dist_sq > 0:
                min_overlap = dist_sq
                push_dx = -dx  # Push in opposite direction
                push_dy = -dy
                collision = True
    
    # If collision detected, push players apart
    if collision:
        # Normalize push vector
        push_dist = math.sqrt(push_dx * push_dx + push_dy * push_dy)
        if push_dist > 0:
            push_dx /= push_dist
            push_dy /= push_dist
        
        # Push each player half the distance
        push_amount = math.sqrt(min_overlap) / 2 + 1  # Add small buffer
        player1.x += push_dx * push_amount
        player1.y += push_dy * push_amount
        player2.x -= push_dx * push_amount
        player2.y -= push_dy * push_amount
        
        return True
    
    return False

def _point_in_polygon(px, py, corners):
    """Check if point (px, py) is inside polygon using ray casting."""
    n = len(corners)
    inside = False
    j = n - 1
    
    for i in range(n):
        xi, yi = corners[i]
        xj, yj = corners[j]
        
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside
