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

def test_goal_detection():
    from physics import check_goal
    from entities import Ball
    from config import FIELD_X, FIELD_Y, FIELD_HEIGHT, GOAL_WIDTH, BALL_RADIUS
    
    # Ball in left goal
    goal_top = FIELD_Y + (FIELD_HEIGHT - GOAL_WIDTH) // 2
    ball = Ball(x=FIELD_X - BALL_RADIUS - 5, y=goal_top + 10, vx=-5, vy=0)
    result = check_goal(ball)
    assert result == "left"
    
    # Ball in right goal
    from config import FIELD_WIDTH
    ball2 = Ball(x=FIELD_X + FIELD_WIDTH + BALL_RADIUS + 5, y=goal_top + 10, vx=5, vy=0)
    result2 = check_goal(ball2)
    assert result2 == "right"
    
    # Ball not in goal
    ball3 = Ball(x=FIELD_X + 100, y=FIELD_Y + 100)
    result3 = check_goal(ball3)
    assert result3 is None
