"""
2048 Game Implementation as a Gymnasium Environment
This module provides a complete 2048 game engine that follows the Gymnasium interface
for use in reinforcement learning tasks.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class Game2048(gym.Env):
    """
    2048 Game Environment compatible with Gymnasium API.
    
    The environment simulates the 2048 game where:
    - Actions: 0=up, 1=down, 2=left, 3=right
    - Observation: 4x4 grid with tile values
    - Reward: Sum of merged tiles in each step
    - Episode termination: When no more moves are possible
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        """
        Initialize the 2048 game environment.
        
        Args:
            render_mode: Optional render mode ('human' or None)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.render_mode = render_mode
        self._rng = np.random.RandomState(seed)
        
        # Game grid: 4x4
        self.grid: np.ndarray = np.zeros((4, 4), dtype=np.int32)
        self.score: int = 0
        self.moves_made: int = 0
        
        # Action space: 4 directions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 4x4 grid with values 0-16 (2^0 to 2^16)
        # Values represent log2 of the actual tile value (0 means empty, 1 means 2, 2 means 4, etc.)
        self.observation_space = spaces.Box(
            low=0,
            high=16,
            shape=(4, 4),
            dtype=np.int32
        )
        
        self._initialize_game()

    def _initialize_game(self) -> None:
        """Initialize the game board with two random tiles."""
        self.grid = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.moves_made = 0
        
        # Add two initial tiles
        self._spawn_tile()
        self._spawn_tile()

    def _spawn_tile(self) -> None:
        """Spawn a new tile (2 or 4) at a random empty position."""
        empty_cells = np.argwhere(self.grid == 0)
        
        if len(empty_cells) == 0:
            return  # No empty cells
        
        # Choose random empty cell
        idx = self._rng.choice(len(empty_cells))
        row, col = empty_cells[idx]
        
        # 90% chance for 2, 10% chance for 4
        value = 1 if self._rng.random() < 0.9 else 2  # 1 means 2^1=2, 2 means 2^2=4
        self.grid[row, col] = value

    def _move_tiles(self, direction: int) -> int:
        """
        Move tiles in the specified direction and merge them.
        
        Args:
            direction: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            Reward (sum of merged tiles)
        """
        grid = self.grid.copy()
        reward = 0
        
        if direction == 2:  # left
            grid, reward = self._move_left(grid)
        elif direction == 3:  # right
            grid, reward = self._move_right(grid)
        elif direction == 0:  # up
            grid, reward = self._move_up(grid)
        elif direction == 1:  # down
            grid, reward = self._move_down(grid)
        
        self.grid = grid
        self.score += reward
        return reward

    def _move_left(self, grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """Move and merge tiles to the left."""
        reward = 0
        for i in range(4):
            row = grid[i, :]
            row, row_reward = self._merge_line(row)
            grid[i, :] = row
            reward += row_reward
        return grid, reward

    def _move_right(self, grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """Move and merge tiles to the right."""
        reward = 0
        for i in range(4):
            row = grid[i, ::-1]
            row, row_reward = self._merge_line(row)
            grid[i, :] = row[::-1]
            reward += row_reward
        return grid, reward

    def _move_up(self, grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """Move and merge tiles upward."""
        reward = 0
        for j in range(4):
            col = grid[:, j]
            col, col_reward = self._merge_line(col)
            grid[:, j] = col
            reward += col_reward
        return grid, reward

    def _move_down(self, grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """Move and merge tiles downward."""
        reward = 0
        for j in range(4):
            col = grid[:, j]
            col = col[::-1]
            col, col_reward = self._merge_line(col)
            col = col[::-1]
            grid[:, j] = col
            reward += col_reward
        return grid, reward

    def _merge_line(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Merge a line (row or column) toward the beginning.
        
        Args:
            line: 1D array of tile values
            
        Returns:
            Tuple of (merged_line, reward)
        """
        reward = 0
        new_line = np.array([x for x in line if x != 0], dtype=np.int32)
        
        # Merge adjacent equal tiles
        merged = []
        i = 0
        while i < len(new_line):
            if i + 1 < len(new_line) and new_line[i] == new_line[i + 1]:
                # Merge two tiles
                merged_value = new_line[i] + 1
                merged.append(merged_value)
                reward += 2 ** merged_value  # Add the actual value as reward
                i += 2
            else:
                merged.append(new_line[i])
                i += 1
        
        # Pad with zeros
        merged_line = np.array(merged, dtype=np.int32)
        padded_line = np.zeros(4, dtype=np.int32)
        padded_line[:len(merged_line)] = merged_line
        
        return padded_line, reward

    def _is_game_over(self) -> bool:
        """Check if the game is over (no more moves possible)."""
        # Check for empty cells
        if np.any(self.grid == 0):
            return False
        
        # Check if any moves are possible (horizontal or vertical merges)
        for i in range(4):
            for j in range(4):
                current = self.grid[i, j]
                
                # Check right neighbor
                if j < 3 and self.grid[i, j + 1] == current:
                    return False
                
                # Check bottom neighbor
                if i < 3 and self.grid[i + 1, j] == current:
                    return False
        
        return True

    def _has_valid_move(self) -> bool:
        """Check if any valid move exists (grid changed after move)."""
        for action in range(4):
            grid_copy = self.grid.copy()
            
            if action == 2:  # left
                temp_grid, _ = self._move_left(grid_copy)
            elif action == 3:  # right
                temp_grid, _ = self._move_right(grid_copy)
            elif action == 0:  # up
                temp_grid, _ = self._move_up(grid_copy)
            elif action == 1:  # down
                temp_grid, _ = self._move_down(grid_copy)
            
            if not np.array_equal(grid_copy, temp_grid):
                return True
        
        return False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Invalid action type: {type(action)}")
        
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action: {action}")
        
        # Store old grid to check if move was valid
        old_grid = self.grid.copy()
        
        # Move tiles and get reward
        reward = self._move_tiles(action)
        
        # Check if the move was valid (grid changed)
        grid_changed = not np.array_equal(old_grid, self.grid)
        
        # If grid changed, spawn a new tile
        if grid_changed:
            self._spawn_tile()
            self.moves_made += 1
        
        # Check if game is over
        terminated = self._is_game_over()
        
        # Truncated is False (we don't have a time limit)
        truncated = False
        
        # Info dictionary
        info = {
            "score": self.score,
            "moves": self.moves_made,
            "grid_changed": grid_changed,
            "game_over": terminated
        }
        
        # Observation: return the grid as is (values are log2 of tile values)
        observation = self.grid.astype(np.int32)
        
        return observation, float(reward), terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional configuration dictionary
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            self._rng.seed(seed)
        
        self._initialize_game()
        
        observation = self.grid.astype(np.int32)
        info = {
            "score": self.score,
            "moves": self.moves_made,
            "game_over": False
        }
        
        return observation, info

    def render(self) -> Optional[str]:
        """
        Render the current game state.
        
        Returns:
            String representation of the grid if render_mode is 'human'
        """
        if self.render_mode == "human":
            output = self._render_string()
            print(output)
            return output
        
        return None

    def _render_string(self) -> str:
        """Generate string representation of the game state."""
        lines = ["\n2048 Game State:"]
        lines.append(f"Score: {self.score} | Moves: {self.moves_made}")
        lines.append("-" * 25)
        
        for i in range(4):
            row_str = ""
            for j in range(4):
                value = self.grid[i, j]
                if value == 0:
                    tile_value = "."
                else:
                    tile_value = str(2 ** value)
                row_str += f"{tile_value:>5}"
            lines.append(row_str)
        
        lines.append("-" * 25)
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up resources."""
        pass

    def get_grid_actual_values(self) -> np.ndarray:
        """
        Get the actual tile values (not log2 values).
        
        Returns:
            4x4 array with actual 2048 game values
        """
        # Convert log2 values to actual values
        actual_grid = np.zeros_like(self.grid, dtype=np.int32)
        mask = self.grid > 0
        actual_grid[mask] = 2 ** self.grid[mask]
        return actual_grid


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = Game2048(render_mode="human")
    
    # Reset and get initial observation
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    env.render()
    
    # Take some random steps
    print("\nTaking 10 random steps:")
    for step_num in range(10):
        action = env.action_space.sample()
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        print(f"\nStep {step_num + 1}: Action = {action_names[action]}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}, Score: {info['score']}, Grid Changed: {info['grid_changed']}")
        env.render()
        
        if terminated:
            print("Game Over!")
            break
    
    print(f"\nActual grid values:\n{env.get_grid_actual_values()}")
