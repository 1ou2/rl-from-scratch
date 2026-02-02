"""
2048 Game - Graphical Interface using Pygame
Play the 2048 game with keyboard controls to test the environment.
"""

import pygame
import sys
import numpy as np
from game_2048 import Game2048


# Color scheme matching the original 2048 game
COLORS = {
    'background': (187, 173, 160),
    'grid': (205, 193, 180),
    'empty': (205, 193, 180),
    'text_dark': (119, 110, 101),
    'text_light': (249, 246, 242),
    'game_bg': (250, 248, 239),
    # Tile colors (based on tile value)
    'tiles': {
        0: (205, 193, 180),      # empty
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
        4096: (60, 58, 50),
        8192: (60, 58, 50),
    }
}


class Game2048GUI:
    """Graphical interface for playing 2048 game."""
    
    def __init__(self, width=500, height=600):
        """
        Initialize the game GUI.
        
        Args:
            width: Window width
            height: Window height
        """
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2048 Game")
        
        # Game parameters
        self.grid_size = 4
        self.tile_size = 100
        self.tile_margin = 15
        self.grid_padding = 20
        
        # Calculate grid position (centered)
        grid_width = self.tile_size * 4 + self.tile_margin * 5
        self.grid_x = (width - grid_width) // 2
        self.grid_y = 150
        
        # Fonts
        self.title_font = pygame.font.Font(None, 80)
        self.score_font = pygame.font.Font(None, 36)
        self.tile_font_large = pygame.font.Font(None, 60)
        self.tile_font_medium = pygame.font.Font(None, 50)
        self.tile_font_small = pygame.font.Font(None, 40)
        self.info_font = pygame.font.Font(None, 24)
        
        # Initialize game environment
        self.env = Game2048()
        self.obs, self.info = self.env.reset()
        self.game_over = False
        
        # Animation
        self.clock = pygame.time.Clock()
        self.fps = 60
        
    def _get_tile_font(self, value):
        """Get appropriate font size based on tile value."""
        if value < 100:
            return self.tile_font_large
        elif value < 1000:
            return self.tile_font_medium
        else:
            return self.tile_font_small
    
    def _get_tile_color(self, value):
        """Get color for a tile based on its value."""
        if value in COLORS['tiles']:
            return COLORS['tiles'][value]
        else:
            return COLORS['tiles'][8192]  # Default for very large values
    
    def _get_text_color(self, value):
        """Get text color based on tile value (dark for light tiles, light for dark tiles)."""
        if value <= 4:
            return COLORS['text_dark']
        else:
            return COLORS['text_light']
    
    def draw_tile(self, screen, value, x, y, size):
        """
        Draw a single tile.
        
        Args:
            screen: Pygame screen surface
            value: Tile value (actual value, not log2)
            x, y: Position
            size: Tile size
        """
        # Draw tile background with rounded corners
        tile_color = self._get_tile_color(value)
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(screen, tile_color, rect, border_radius=8)
        
        # Draw tile value
        if value > 0:
            font = self._get_tile_font(value)
            text_color = self._get_text_color(value)
            text = font.render(str(value), True, text_color)
            text_rect = text.get_rect(center=(x + size // 2, y + size // 2))
            screen.blit(text, text_rect)
    
    def draw_grid(self):
        """Draw the game grid with all tiles."""
        # Draw grid background
        grid_width = self.tile_size * 4 + self.tile_margin * 5
        grid_height = self.tile_size * 4 + self.tile_margin * 5
        grid_rect = pygame.Rect(
            self.grid_x - self.tile_margin,
            self.grid_y - self.tile_margin,
            grid_width,
            grid_height
        )
        pygame.draw.rect(self.screen, COLORS['background'], grid_rect, border_radius=10)
        
        # Get actual tile values
        actual_grid = self.env.get_grid_actual_values()
        
        # Draw each tile
        for i in range(4):
            for j in range(4):
                x = self.grid_x + j * (self.tile_size + self.tile_margin)
                y = self.grid_y + i * (self.tile_size + self.tile_margin)
                value = actual_grid[i, j]
                self.draw_tile(self.screen, value, x, y, self.tile_size)
    
    def draw_header(self):
        """Draw the header with title and score."""
        # Title
        title_text = self.title_font.render("2048", True, COLORS['text_dark'])
        self.screen.blit(title_text, (30, 30))
        
        # Score box
        score_box_width = 150
        score_box_height = 70
        score_box_x = self.width - score_box_width - 30
        score_box_y = 30
        
        # Draw score background
        score_rect = pygame.Rect(score_box_x, score_box_y, score_box_width, score_box_height)
        pygame.draw.rect(self.screen, COLORS['background'], score_rect, border_radius=8)
        
        # Score label
        score_label = self.info_font.render("SCORE", True, COLORS['text_light'])
        score_label_rect = score_label.get_rect(center=(score_box_x + score_box_width // 2, score_box_y + 20))
        self.screen.blit(score_label, score_label_rect)
        
        # Score value
        score_value = self.score_font.render(str(self.info['score']), True, (255, 255, 255))
        score_value_rect = score_value.get_rect(center=(score_box_x + score_box_width // 2, score_box_y + 48))
        self.screen.blit(score_value, score_value_rect)
    
    def draw_instructions(self):
        """Draw instructions at the bottom."""
        instructions = [
            "Use Arrow Keys to move tiles",
            "Press R to restart | ESC to quit"
        ]
        
        y_offset = self.height - 60
        for instruction in instructions:
            text = self.info_font.render(instruction, True, COLORS['text_dark'])
            text_rect = text.get_rect(center=(self.width // 2, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 25
    
    def draw_game_over(self):
        """Draw game over overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill((238, 228, 218))
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        game_over_text = self.title_font.render("Game Over!", True, COLORS['text_dark'])
        game_over_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 40))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Final score
        final_score_text = self.score_font.render(f"Final Score: {self.info['score']}", True, COLORS['text_dark'])
        final_score_rect = final_score_text.get_rect(center=(self.width // 2, self.height // 2 + 20))
        self.screen.blit(final_score_text, final_score_rect)
        
        # Restart instruction
        restart_text = self.info_font.render("Press R to restart", True, COLORS['text_dark'])
        restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 60))
        self.screen.blit(restart_text, restart_rect)
    
    def draw(self):
        """Draw the entire game state."""
        # Clear screen
        self.screen.fill(COLORS['game_bg'])
        
        # Draw components
        self.draw_header()
        self.draw_grid()
        self.draw_instructions()
        
        # Draw game over overlay if needed
        if self.game_over:
            self.draw_game_over()
        
        pygame.display.flip()
    
    def handle_input(self, event):
        """
        Handle keyboard input.
        
        Args:
            event: Pygame event
        """
        if event.type == pygame.QUIT:
            return False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            
            if event.key == pygame.K_r:
                # Restart game
                self.obs, self.info = self.env.reset()
                self.game_over = False
                return True
            
            if not self.game_over:
                # Map arrow keys to actions
                action = None
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                
                # Execute action
                if action is not None:
                    self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                    
                    if terminated:
                        self.game_over = True
        
        return True
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if not self.handle_input(event):
                    running = False
            
            # Draw
            self.draw()
            
            # Control frame rate
            self.clock.tick(self.fps)
        
        pygame.quit()
        sys.exit()


def main():
    """Main entry point."""
    game = Game2048GUI(width=500, height=600)
    game.run()


if __name__ == "__main__":
    main()
