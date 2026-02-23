import pygame
import sys
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from entities import Player, Ball

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2D Football Game")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Create players
        self.player1 = Player(
            x=SCREEN_WIDTH // 2 - 200,
            y=SCREEN_HEIGHT // 2,
            angle=0,
            color=(255, 0, 0)  # Red
        )
        self.player2 = Player(
            x=SCREEN_WIDTH // 2 + 200,
            y=SCREEN_HEIGHT // 2,
            angle=180,
            color=(0, 0, 255)  # Blue
        )
        
        # Create ball
        self.ball = Ball(x=SCREEN_WIDTH // 2, y=SCREEN_HEIGHT // 2, vx=0, vy=0)
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def update(self):
        from physics import check_ball_wall_collision, check_player_wall_collision
        
        # Update ball
        self.ball.update()
        check_ball_wall_collision(self.ball)
        
        # Keep players in bounds
        check_player_wall_collision(self.player1)
        check_player_wall_collision(self.player2)
    
    def render(self):
        from renderer import render_field, render_player, render_ball
        render_field(self.screen)
        render_ball(self.screen, self.ball)
        render_player(self.screen, self.player1)
        render_player(self.screen, self.player2)
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
