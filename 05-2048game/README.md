# 2048 Game - Gymnasium Environment

A complete implementation of the 2048 game as a Gymnasium environment for reinforcement learning tasks.

## Overview

This environment implements the classic 2048 game following the Gymnasium interface, making it compatible with most RL algorithms and frameworks.

## Game Mechanics

- **Grid**: 4x4 board
- **Tiles**: Values represented as log₂ (0 = empty, 1 = 2, 2 = 4, 3 = 8, ..., 16 = 2^16)
- **Moves**: 4 actions (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
- **Spawning**: New tiles (2 or 4) spawn after each valid move
- **Merging**: Tiles with same value merge when moved together
- **Termination**: Game ends when no more moves are possible

## Interface

### Environment Setup

```python
import gymnasium as gym
from game_2048 import Game2048

# Create environment
env = Game2048(render_mode=None)
obs, info = env.reset(seed=42)
```

### Step Function

```python
action = env.action_space.sample()  # Random action (0-3)
obs, reward, terminated, truncated, info = env.step(action)
```

**Returns:**
- `obs` (np.ndarray): 4x4 grid with log₂ tile values
- `reward` (float): Sum of tiles merged in this step
- `terminated` (bool): True when game is over
- `truncated` (bool): False (no time limit)
- `info` (dict): Additional information including score and moves count

### Action Space

- `0` - Move UP
- `1` - Move DOWN
- `2` - Move LEFT
- `3` - Move RIGHT

### Observation Space

- **Type**: Box
- **Shape**: (4, 4)
- **Values**: 0-16 (representing log₂ of actual tile values)
  - 0 = empty
  - 1 = tile value 2
  - 2 = tile value 4
  - n = tile value 2^n

### Reward

The reward for each step is the **sum of all tiles created by merging** in that move:
- Merging two 2s (log₂ value 1) creates a 4 (value 4 in reward)
- Merging two 4s (log₂ value 2) creates an 8 (value 8 in reward)
- And so on...

Invalid moves (no grid change) result in 0 reward and no new tile spawn.

## Utilities

### Get Actual Grid Values

Convert log₂ representation to actual tile values:

```python
actual_grid = env.get_grid_actual_values()
# actual_grid contains values like 2, 4, 8, 16, etc.
```

### Render

Display the current game state:

```python
env.render()  # Prints formatted grid
```

### Info Dictionary

After each step, `info` contains:
- `score`: Total score (sum of all merged tiles)
- `moves`: Number of moves made
- `grid_changed`: Whether the last move was valid
- `game_over`: Whether the game is over

## Example Usage

```python
from game_2048 import Game2048

# Initialize
env = Game2048(render_mode="human")
obs, info = env.reset(seed=42)

# Play a few steps
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward: {reward}, Score: {info['score']}")
    
    if terminated:
        print("Game Over!")
        break

env.close()
```

## Integration with RL Algorithms

This environment is compatible with:
- Stable-baselines3
- Ray RLlib
- PyTorch RL frameworks
- Any framework supporting Gymnasium

### Example with Stable-baselines3

```python
from stable_baselines3 import DQN
from game_2048 import Game2048

env = Game2048()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

obs, info = env.reset()
for _ in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, info = env.reset()
```

## Design Notes

- **State Representation**: The grid uses log₂ values to keep values small and convenient for neural networks
- **Reward Shaping**: Rewards are based on tile values created to encourage merging high-value tiles
- **Invalid Moves**: Attempting an impossible move results in 0 reward but doesn't penalize the agent
- **Game Over Detection**: Efficient checking considers both empty cells and possible merges
- **Deterministic Randomness**: All randomness can be controlled via seed for reproducibility

## Playing the Game

A graphical interface is provided to manually play and test the environment:

```bash
python play_2048.py
```

**Controls:**
- **Arrow Keys**: Move tiles (↑ ↓ ← →)
- **R**: Restart game
- **ESC**: Quit

The GUI matches the original 2048 game design with proper colors, rounded tiles, and score display.

## Requirements

- `gymnasium >= 0.27.0`
- `numpy >= 1.20.0`
- `pygame >= 2.0.0` (for playing the game graphically)

## Files

- `game_2048.py`: Main environment implementation
- `play_2048.py`: Pygame-based graphical interface for playing the game
- `README.md`: Documentation
