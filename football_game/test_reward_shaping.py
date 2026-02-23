"""Test reward shaping manually."""
from football_env import FootballEnv
from config import FIELD_X, FIELD_WIDTH
import numpy as np

env = FootballEnv(opponent_type="stationary")
obs, info = env.reset()

print("Testing reward shaping...")
print(f"Initial ball position: ({env.state.ball.x:.1f}, {env.state.ball.y:.1f})")
print(f"Opponent half starts at x = {FIELD_X + FIELD_WIDTH / 2:.1f}")

# Try moving toward the ball and kicking
total_reward = 0
for step in range(100):
    # Simple heuristic: move toward ball
    dx = env.state.ball.x - env.state.player1.x
    dy = env.state.ball.y - env.state.player1.y
    angle_to_ball = np.degrees(np.arctan2(dy, dx))
    
    # Get current player angle
    player_angle = env.state.player1.angle
    angle_diff = (angle_to_ball - player_angle + 180) % 360 - 180
    
    if abs(angle_diff) > 10:
        action = 2 if angle_diff > 0 else 3  # Rotate toward ball
    else:
        action = 0  # Move forward
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if reward != 0:
        print(f"Step {step}: reward={reward:.3f}, ball=({env.state.ball.x:.1f}, {env.state.ball.y:.1f}), "
              f"shaped={info.get('shaped_reward', 0):.3f}")
    
    if terminated or truncated:
        print(f"\nEpisode ended after {step+1} steps")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Final score: {info['score1']}-{info['score2']}")
        break

env.close()
print("\nTest complete!")
