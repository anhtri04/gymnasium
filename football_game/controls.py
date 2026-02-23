import pygame
from config import PLAYER_ROTATION_SPEED, PLAYER_SPEED
from physics import kick_ball, dribble_ball, is_ball_in_kick_arc

def handle_player1_controls(keys, player, ball):
    """Handle WASD controls for player 1."""
    # Rotation
    if keys[pygame.K_a]:
        player.rotate(-PLAYER_ROTATION_SPEED)
    if keys[pygame.K_d]:
        player.rotate(PLAYER_ROTATION_SPEED)
    
    # Movement
    if keys[pygame.K_w]:
        player.move_forward(PLAYER_SPEED)
        # Check if dribbling
        if is_ball_in_kick_arc(player, ball):
            dribble_ball(player, ball)
    if keys[pygame.K_s]:
        player.move_backward(PLAYER_SPEED)
    
    # Kick
    if keys[pygame.K_SPACE]:
        if is_ball_in_kick_arc(player, ball):
            kick_ball(player, ball)

def handle_player2_controls(keys, player, ball):
    """Handle Arrow key controls for player 2."""
    # Rotation
    if keys[pygame.K_LEFT]:
        player.rotate(-PLAYER_ROTATION_SPEED)
    if keys[pygame.K_RIGHT]:
        player.rotate(PLAYER_ROTATION_SPEED)
    
    # Movement
    if keys[pygame.K_UP]:
        player.move_forward(PLAYER_SPEED)
        # Check if dribbling
        if is_ball_in_kick_arc(player, ball):
            dribble_ball(player, ball)
    if keys[pygame.K_DOWN]:
        player.move_backward(PLAYER_SPEED)
    
    # Kick
    if keys[pygame.K_RETURN]:
        if is_ball_in_kick_arc(player, ball):
            kick_ball(player, ball)
