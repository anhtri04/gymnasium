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

def render_scoreboard(screen, score1, score2, episode_time):
    """Render scoreboard and timer at top of screen."""
    from config import SCREEN_WIDTH, WHITE, EPISODE_TIME_LIMIT
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
    
    # Draw timer below score
    time_remaining = max(0, EPISODE_TIME_LIMIT - int(episode_time))
    minutes = time_remaining // 60
    seconds = time_remaining % 60
    timer_text = f"Time: {minutes}:{seconds:02d}"
    timer_surface = small_font.render(timer_text, True, WHITE)
    timer_rect = timer_surface.get_rect(center=(SCREEN_WIDTH // 2, 65))
    screen.blit(timer_surface, timer_rect)
