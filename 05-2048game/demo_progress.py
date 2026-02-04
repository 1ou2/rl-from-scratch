"""
Demo Script: Showcase Model Progress with Multiple Checkpoints
Renders gameplay from different training checkpoints to show learning progression
"""

import pygame
import sys
import numpy as np
import torch
import glob
import os
from typing import List, Optional
import time

from game_2048 import Game2048
from neural_network import DQN2048
from play_2048 import Game2048GUI, COLORS


class ModelProgressDemo:
    """
    Demonstrates model progress by playing games with different checkpoints.
    Can save frames for video creation.
    """
    
    def __init__(self, checkpoints: List[str], episodes_per_checkpoint: int = 3,
                 delay_between_moves: int = 300, save_video_frames: bool = False,
                 output_dir: str = "demo_frames"):
        """
        Initialize the demo.
        
        Args:
            checkpoints: List of checkpoint paths
            episodes_per_checkpoint: Number of games to play per checkpoint
            delay_between_moves: Milliseconds between moves (for visibility)
            save_video_frames: Whether to save frames for video creation
            output_dir: Directory to save frames
        """
        self.checkpoints = sorted(checkpoints, key=self._extract_episode_number)
        self.episodes_per_checkpoint = episodes_per_checkpoint
        self.delay_between_moves = delay_between_moves
        self.save_video_frames = save_video_frames
        self.output_dir = output_dir
        
        if save_video_frames:
            os.makedirs(output_dir, exist_ok=True)
            self.frame_counter = 0
        
        # Initialize pygame
        pygame.init()
        self.width = 500
        self.height = 650  # Extra space for checkpoint info
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("2048 Model Progress Demo")
        
        # GUI components (reuse from play_2048.py)
        self.gui = Game2048GUI(width=self.width, height=self.height)
        
        # Device for model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Stats tracking
        self.current_checkpoint_name = ""
        self.current_episode = 0
        self.checkpoint_stats = []
        
    def _extract_episode_number(self, checkpoint_path: str) -> int:
        """Extract episode number from checkpoint filename."""
        try:
            # Extract number from format like "dqn_2048_ep500.pth"
            basename = os.path.basename(checkpoint_path)
            if 'ep' in basename:
                ep_str = basename.split('ep')[1].split('.')[0]
                return int(ep_str)
            return 0
        except:
            return 0
    
    def load_model(self, checkpoint_path: str) -> DQN2048:
        """Load a model from checkpoint."""
        model = DQN2048().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['policy_net_state_dict'])
        model.eval()
        
        # Get episode number for display
        episode_num = self._extract_episode_number(checkpoint_path)
        self.current_checkpoint_name = f"Episode {episode_num}"
        
        return model, episode_num
    
    def select_action(self, model: DQN2048, state: np.ndarray) -> int:
        """Select action using the model (greedy)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = model(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def draw_checkpoint_info(self, episode_num: int, game_num: int, max_tile: int):
        """Draw checkpoint information at the top."""
        # Background for info
        info_rect = pygame.Rect(0, 0, self.width, 80)
        pygame.draw.rect(self.screen, (250, 248, 239), info_rect)
        
        # Checkpoint info - left side
        checkpoint_font = pygame.font.Font(None, 28)
        checkpoint_text = checkpoint_font.render(
            f"Checkpoint: Episode {episode_num}", 
            True, COLORS['text_dark']
        )
        self.screen.blit(checkpoint_text, (15, 10))
        
        # Game number and max tile - left side, second line
        game_font = pygame.font.Font(None, 22)
        game_text = game_font.render(
            f"Game {game_num}/{self.episodes_per_checkpoint} | Max Tile: {max_tile}",
            True, COLORS['text_dark']
        )
        self.screen.blit(game_text, (15, 42))
        
        # Divider line
        pygame.draw.line(self.screen, COLORS['background'], (0, 80), (self.width, 80), 2)
    
    def play_game(self, model: DQN2048, episode_num: int, game_num: int) -> dict:
        """
        Play one game with the given model.
        
        Returns:
            Game statistics
        """
        env = Game2048()
        state, info = env.reset()
        done = False
        truncated = False
        moves = 0
        max_tile = 0
        
        # Initialize GUI for this game
        self.gui.env = env
        self.gui.obs = state
        self.gui.info = info
        self.gui.game_over = False
        
        while not (done or truncated):
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None  # Signal to stop demo
            
            # Select and execute action
            action = self.select_action(model, state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update GUI state
            self.gui.obs = next_state
            self.gui.info = info
            
            # Calculate max tile
            actual_grid = env.get_grid_actual_values()
            max_tile = int(actual_grid.max())
            
            # Draw
            self.screen.fill(COLORS['game_bg'])
            self.draw_checkpoint_info(episode_num, game_num, max_tile)
            
            # Draw grid only (skip overlapping header with title)
            original_y = self.gui.grid_y
            self.gui.grid_y = 120  # Lower position to avoid checkpoint info
            self.gui.draw_grid()
            self.gui.grid_y = original_y
            
            pygame.display.flip()
            
            # Save frame if recording
            if self.save_video_frames:
                self._save_frame()
            
            # Delay for visibility
            pygame.time.wait(self.delay_between_moves)
            
            state = next_state
            moves += 1
        
        # Show final state briefly
        self.gui.game_over = True
        self.screen.fill(COLORS['game_bg'])
        self.draw_checkpoint_info(episode_num, game_num, max_tile)
        
        original_y = self.gui.grid_y
        self.gui.grid_y = 120  # Lower position to avoid checkpoint info
        self.gui.draw_grid()
        self.gui.draw_game_over()
        self.gui.grid_y = original_y
        
        pygame.display.flip()
        
        if self.save_video_frames:
            for _ in range(30):  # Hold final frame
                self._save_frame()
        
        pygame.time.wait(2000)  # Show final state for 2 seconds
        
        return {
            'score': info['score'],
            'max_tile': max_tile,
            'moves': moves
        }
    
    def _save_frame(self):
        """Save current screen as a frame."""
        if self.save_video_frames:
            filename = os.path.join(self.output_dir, f"frame_{self.frame_counter:06d}.png")
            pygame.image.save(self.screen, filename)
            self.frame_counter += 1
    
    def run_demo(self):
        """Run the full demo showing progression through checkpoints."""
        print(f"\n{'='*60}")
        print(f"2048 Model Progress Demo")
        print(f"{'='*60}")
        print(f"Checkpoints to demo: {len(self.checkpoints)}")
        print(f"Games per checkpoint: {self.episodes_per_checkpoint}")
        print(f"Delay between moves: {self.delay_between_moves}ms")
        print(f"Save video frames: {self.save_video_frames}")
        if self.save_video_frames:
            print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for i, checkpoint_path in enumerate(self.checkpoints):
            print(f"\n[{i+1}/{len(self.checkpoints)}] Loading checkpoint: {checkpoint_path}")
            model, episode_num = self.load_model(checkpoint_path)
            
            checkpoint_scores = []
            checkpoint_max_tiles = []
            
            for game_num in range(1, self.episodes_per_checkpoint + 1):
                print(f"  Playing game {game_num}/{self.episodes_per_checkpoint}...")
                
                stats = self.play_game(model, episode_num, game_num)
                
                if stats is None:  # User pressed ESC
                    print("\nDemo stopped by user.")
                    self._create_video_instructions()
                    pygame.quit()
                    return
                
                checkpoint_scores.append(stats['score'])
                checkpoint_max_tiles.append(stats['max_tile'])
                
                print(f"    Score: {stats['score']}, Max Tile: {stats['max_tile']}, Moves: {stats['moves']}")
            
            # Print checkpoint summary
            avg_score = np.mean(checkpoint_scores)
            avg_max_tile = np.mean(checkpoint_max_tiles)
            best_max_tile = max(checkpoint_max_tiles)
            
            print(f"\n  Checkpoint Summary:")
            print(f"    Avg Score: {avg_score:.2f}")
            print(f"    Avg Max Tile: {avg_max_tile:.0f}")
            print(f"    Best Max Tile: {best_max_tile}")
            
            self.checkpoint_stats.append({
                'episode': episode_num,
                'avg_score': avg_score,
                'avg_max_tile': avg_max_tile,
                'best_max_tile': best_max_tile
            })
        
        print(f"\n{'='*60}")
        print("Demo Complete!")
        print(f"{'='*60}\n")
        
        # Print overall summary
        self._print_summary()
        
        if self.save_video_frames:
            self._create_video_instructions()
        
        # Wait for user to close
        print("\nPress ESC or close window to exit...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False
            pygame.time.wait(100)
        
        pygame.quit()
    
    def _print_summary(self):
        """Print summary of all checkpoints."""
        print("\nOverall Progress Summary:")
        print("-" * 60)
        print(f"{'Episode':>10} | {'Avg Score':>12} | {'Avg Tile':>10} | {'Best Tile':>10}")
        print("-" * 60)
        
        for stat in self.checkpoint_stats:
            print(f"{stat['episode']:>10} | {stat['avg_score']:>12.2f} | "
                  f"{stat['avg_max_tile']:>10.0f} | {stat['best_max_tile']:>10}")
        
        print("-" * 60)
    
    def _create_video_instructions(self):
        """Print instructions for creating video from saved frames."""
        if not self.save_video_frames:
            return
        
        print(f"\n{'='*60}")
        print("Video Creation Instructions")
        print(f"{'='*60}")
        print(f"Frames saved to: {self.output_dir}/")
        print(f"Total frames: {self.frame_counter}")
        print("\nTo create a video, use ffmpeg:")
        print(f"\nffmpeg -framerate 30 -i {self.output_dir}/frame_%06d.png \\")
        print(f"       -c:v libx264 -pix_fmt yuv420p \\")
        print(f"       -crf 18 \\")
        print(f"       model_progress.mp4")
        print(f"\n{'='*60}\n")


def find_all_checkpoints(models_dir: str = "models") -> List[str]:
    """Find all checkpoint files."""
    checkpoints = glob.glob(os.path.join(models_dir, "dqn_2048_ep*.pth"))
    return sorted(checkpoints, key=lambda x: int(x.split('ep')[1].split('.')[0]))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo 2048 Model Progress')
    parser.add_argument('--checkpoints', nargs='+', default=None,
                        help='Specific checkpoints to demo (e.g., models/dqn_2048_ep500.pth models/dqn_2048_ep1000.pth)')
    parser.add_argument('--all', action='store_true',
                        help='Demo all available checkpoints')
    parser.add_argument('--interval', type=int, default=1,
                        help='Use every Nth checkpoint (default: 1, use all)')
    parser.add_argument('--games', type=int, default=3,
                        help='Number of games per checkpoint (default: 3)')
    parser.add_argument('--delay', type=int, default=300,
                        help='Delay between moves in ms (default: 300)')
    parser.add_argument('--record', action='store_true',
                        help='Save frames for video creation')
    parser.add_argument('--output-dir', type=str, default='demo_frames',
                        help='Directory for saved frames (default: demo_frames)')
    
    args = parser.parse_args()
    
    # Determine which checkpoints to use
    if args.checkpoints:
        checkpoints = args.checkpoints
    elif args.all:
        checkpoints = find_all_checkpoints()
        if args.interval > 1:
            checkpoints = checkpoints[::args.interval]
    else:
        # Default: find key checkpoints
        all_checkpoints = find_all_checkpoints()
        if len(all_checkpoints) == 0:
            print("Error: No checkpoints found in models/")
            print("Train a model first: python train_2048.py")
            return
        
        # Select evenly spaced checkpoints (max 6)
        if len(all_checkpoints) <= 6:
            checkpoints = all_checkpoints
        else:
            indices = np.linspace(0, len(all_checkpoints) - 1, 6, dtype=int)
            checkpoints = [all_checkpoints[i] for i in indices]
    
    if len(checkpoints) == 0:
        print("Error: No checkpoints to demo")
        return
    
    print(f"\nSelected checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp}")
    
    # Run demo
    demo = ModelProgressDemo(
        checkpoints=checkpoints,
        episodes_per_checkpoint=args.games,
        delay_between_moves=args.delay,
        save_video_frames=args.record,
        output_dir=args.output_dir
    )
    
    demo.run_demo()


if __name__ == "__main__":
    main()
