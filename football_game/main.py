import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, EPISODE_TIME_LIMIT
from entities import Player, Ball
from state import GameState
from physics import (
    check_ball_wall_collision,
    check_player_wall_collision,
    check_goal,
    check_ball_player_collision,
    check_player_player_collision
)
from controls import handle_player1_controls, handle_player2_controls

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2D Football Game")
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = GameState()
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def update(self, dt):
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Handle player controls
        handle_player1_controls(keys, self.state.player1, self.state.ball)
        handle_player2_controls(keys, self.state.player2, self.state.ball)
        
        # Update ball
        self.state.ball.update()
        check_ball_wall_collision(self.state.ball)
        
        # Check ball-player collisions (ball bounces off players)
        check_ball_player_collision(self.state.ball, self.state.player1)
        check_ball_player_collision(self.state.ball, self.state.player2)
        
        # Check player-player collisions (players block each other)
        check_player_player_collision(self.state.player1, self.state.player2)
        
        # Check for goals
        goal = check_goal(self.state.ball)
        if goal == "left":
            self.state.increment_score(2)
            self.state.reset_positions()
        elif goal == "right":
            self.state.increment_score(1)
            self.state.reset_positions()
        
        # Keep players in bounds
        check_player_wall_collision(self.state.player1)
        check_player_wall_collision(self.state.player2)
        
        # Update timer
        timed_out = self.state.update_episode_time(dt)
        if timed_out:
            self.state.reset_positions()
    
    def render(self):
        from renderer import render_field, render_player, render_ball, render_scoreboard
        render_field(self.screen)
        render_ball(self.screen, self.state.ball)
        render_player(self.screen, self.state.player1)
        render_player(self.screen, self.state.player2)
        render_scoreboard(self.screen, self.state.score1, self.state.score2, self.state.episode_time)
        pygame.display.flip()
    
    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.render()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
