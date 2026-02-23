def test_ball_wall_bounce():
    from physics import check_ball_wall_collision
    from entities import Ball
    from config import FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT
    
    # Ball hitting right wall
    ball = Ball(x=FIELD_X + FIELD_WIDTH + 5, y=FIELD_Y + 100, vx=5, vy=0)
    check_ball_wall_collision(ball)
    assert ball.vx < 0  # Should bounce back
    assert ball.x <= FIELD_X + FIELD_WIDTH  # Should be inside field

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
