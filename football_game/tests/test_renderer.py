def test_field_rect_position():
    from config import FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT
    assert FIELD_X == 100
    assert FIELD_Y == 100
    assert FIELD_WIDTH == 1000
    assert FIELD_HEIGHT == 600

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
