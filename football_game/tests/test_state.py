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

def test_episode_timeout():
    from state import GameState
    from config import EPISODE_TIME_LIMIT
    
    state = GameState()
    
    # Advance time
    timed_out = False
    for i in range(70):  # 70 seconds at 1 sec per call
        timed_out = state.update_episode_time(1.0)
    
    assert timed_out == True
