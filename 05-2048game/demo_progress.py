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
from neural_network_v2 import DQN2048_V2, DQN2048_V2_Large
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
        self.checkpoints = sorted(checkpoints, key=self._extract_step_number)
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
        
    def _extract_step_number(self, checkpoint_path: str) -> int:
        """Extract step number from checkpoint filename."""
        try:
            basename = os.path.basename(checkpoint_path)
            # V2 format: "dqn_2048_v2_step500000.pth"
            if 'step' in basename:
                step_str = basename.split('step')[1].split('.')[0]
                return int(step_str)
            # V1 format: "dqn_2048_ep500.pth"
            elif 'ep' in basename:
                ep_str = basename.split('ep')[1].split('.')[0]
                return int(ep_str)
            # Best model
            elif 'best' in basename:
                return float('inf')  # Sort best to end
            return 0
        except:
            return 0
    
    def load_model(self, checkpoint_path: str) -> DQN2048_V2_Large:
        """Load a model from checkpoint."""
        model = DQN2048_V2_Large().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['policy_net_state_dict'])
        model.eval()
        
        # Get step number for display
        step_num = self._extract_step_number(checkpoint_path)
        basename = os.path.basename(checkpoint_path)
        if 'best' in basename:
            self.current_checkpoint_name = "Best Model"
        else:
            self.current_checkpoint_name = f"Step {step_num:,}"
        
        return model, step_num
    
    def _get_valid_moves(self, env: Game2048) -> list:
        """Get list of valid moves by checking grid change."""
        valid = []
        original_grid = env.grid.copy()
        
        for action in range(4):
            grid_copy = env.grid.copy()
            
            if action == 2:  # left
                temp_grid, _ = env._move_left(grid_copy)
            elif action == 3:  # right
                temp_grid, _ = env._move_right(grid_copy)
            elif action == 0:  # up
                temp_grid, _ = env._move_up(grid_copy)
            elif action == 1:  # down
                temp_grid, _ = env._move_down(grid_copy)
            
            # Compare against original, not grid_copy (which was modified in place)
            if not np.array_equal(original_grid, temp_grid):
                valid.append(action)
        
        return valid
    
    def select_action(self, model: DQN2048_V2_Large, env: Game2048) -> int:
        """Select best valid action using the model."""
        with torch.no_grad():
            # V2 model expects raw grid values (log2), handles encoding internally
            state_tensor = torch.FloatTensor(env.grid).unsqueeze(0).to(self.device)
            q_values = model(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
            
            # Filter for valid moves only
            valid_moves = self._get_valid_moves(env)
            if not valid_moves:
                return 0  # No valid moves, game should be over
            
            valid_q = [(a, q_values_np[a]) for a in valid_moves]
            return max(valid_q, key=lambda x: x[1])[0]
    
    def draw_checkpoint_info(self, step_num: int, game_num: int, max_tile: int):
        """Draw checkpoint information at the top."""
        # Background for info
        info_rect = pygame.Rect(0, 0, self.width, 80)
        pygame.draw.rect(self.screen, (250, 248, 239), info_rect)
        
        # Checkpoint info - left side
        checkpoint_font = pygame.font.Font(None, 28)
        checkpoint_text = checkpoint_font.render(
            f"Checkpoint: {self.current_checkpoint_name}", 
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
    
    def play_game(self, model: DQN2048_V2_Large, step_num: int, game_num: int) -> dict:
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
            
            # Check for valid moves first
            valid_moves = self._get_valid_moves(env)
            if not valid_moves:
                break  # No valid moves, game over
            
            # Select and execute action
            action = self.select_action(model, env)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update GUI state
            self.gui.obs = next_state
            self.gui.info = info
            
            # Calculate max tile
            actual_grid = env.get_grid_actual_values()
            max_tile = int(actual_grid.max())
            
            # Draw
            self.screen.fill(COLORS['game_bg'])
            self.draw_checkpoint_info(step_num, game_num, max_tile)
            
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
        
        # Re-sync final state from environment (ensures we show the actual final grid)
        self.gui.obs = env.grid.astype(np.int32)
        actual_grid = env.get_grid_actual_values()
        max_tile = int(actual_grid.max())
        
        # Show final state briefly
        self.gui.game_over = True
        self.screen.fill(COLORS['game_bg'])
        self.draw_checkpoint_info(step_num, game_num, max_tile)
        
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
            model, step_num = self.load_model(checkpoint_path)
            
            checkpoint_scores = []
            checkpoint_max_tiles = []
            
            for game_num in range(1, self.episodes_per_checkpoint + 1):
                print(f"  Playing game {game_num}/{self.episodes_per_checkpoint}...")
                
                stats = self.play_game(model, step_num, game_num)
                
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
                'step': step_num,
                'name': self.current_checkpoint_name,
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
        print("-" * 70)
        print(f"{'Checkpoint':<20} | {'Avg Score':>12} | {'Avg Tile':>10} | {'Best Tile':>10}")
        print("-" * 70)
        
        for stat in self.checkpoint_stats:
            print(f"{stat['name']:<20} | {stat['avg_score']:>12.2f} | "
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


def find_all_checkpoints(models_dir: str = "../models") -> List[str]:
    """Find all V2 checkpoint files."""
    checkpoints = glob.glob(os.path.join(models_dir, "dqn_2048_v2_step*.pth"))
    # Also include best model if it exists
    best_path = os.path.join(models_dir, "dqn_2048_v2_best.pth")
    if os.path.exists(best_path):
        checkpoints.append(best_path)
    
    def extract_step(path):
        basename = os.path.basename(path)
        if 'best' in basename:
            return float('inf')
        try:
            return int(basename.split('step')[1].split('.')[0])
        except:
            return 0
    
    return sorted(checkpoints, key=extract_step)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo 2048 Model Progress')
    parser.add_argument('--checkpoints', nargs='+', default=None,
                        help='Specific checkpoints to demo (e.g., ../models/dqn_2048_v2_step100000.pth)')
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
            print("Error: No V2 checkpoints found in ../models/")
            print("Train a model first: python train_vectorized_v2.py --large")
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
