import math
from entities import Player, Ball
from config import BALL_FRICTION

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
    player.rotate(280)  # 90 + 280 = 370, should wrap to 10
    assert player.angle == 10

def test_ball_creation():
    ball = Ball(x=500, y=400, vx=5, vy=3)
    assert ball.x == 500
    assert ball.y == 400
    assert ball.vx == 5
    assert ball.vy == 3

def test_ball_update_with_friction():
    ball = Ball(x=100, y=100, vx=10, vy=0)
    ball.update()
    assert ball.vx == 10 * BALL_FRICTION
    assert ball.x == 100 + 10  # Original vx, friction applied after movement
