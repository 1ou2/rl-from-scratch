"""
Improved Vectorized Training Script for 2048 Game - Version 2

Key improvements over V1:
1. Better neural network architecture (CNN + one-hot encoding)
2. Reward shaping for faster learning
3. Faster epsilon decay schedule
4. Prioritized experience replay (optional)
5. Proper hyperparameters tuned for reaching 2048

Usage:
    python train_vectorized_v2.py                          # Train from scratch
    python train_vectorized_v2.py --resume models/xxx.pth  # Resume training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
import os
import logging
import sys
from datetime import datetime

from vectorized_game_2048 import VectorizedGame2048, VectorizedReplayBuffer
from neural_network_v2 import DQN2048_V2, DQN2048_V2_Large


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Setup logging to both console and file.
    
    Args:
        log_dir: Directory to store log files
    
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("train_2048")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers (in case of re-run)
    logger.handlers.clear()
    
    # File handler - logs everything
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    # Console handler - logs everything
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger


# Global logger (initialized in main)
logger: Optional[logging.Logger] = None


def log(message: str):
    """Log a message to both console and file."""
    if logger:
        logger.info(message)
    else:
        print(message)


def compute_reward_shaping(
    states: np.ndarray, 
    next_states: np.ndarray, 
    base_rewards: np.ndarray,
    dones: np.ndarray,
    grid_changed: np.ndarray
) -> np.ndarray:
    """
    Compute shaped rewards to guide learning.
    
    Reward components:
    1. Base merge reward (unchanged)
    2. Empty cells bonus (encourage keeping board open)
    3. Max tile progress bonus (encourage merging to higher tiles)
    4. Corner bonus (reward keeping max tile in corner)
    5. Monotonicity bonus (reward keeping tiles in order)
    
    Args:
        states: Previous states (num_envs, 4, 4)
        next_states: New states (num_envs, 4, 4)
        base_rewards: Original merge rewards (num_envs,)
        dones: Done flags (num_envs,)
        grid_changed: Whether grid changed (num_envs,)
    
    Returns:
        Shaped rewards (num_envs,)
    """
    num_envs = states.shape[0]
    shaped_rewards = base_rewards.astype(np.float32).copy()
    
    # 1. Empty cells bonus (encourage keeping the board open)
    # More empty cells = more freedom to maneuver
    empty_before = np.sum(states == 0, axis=(1, 2))
    empty_after = np.sum(next_states == 0, axis=(1, 2))
    # Small bonus for maintaining empty cells
    empty_bonus = (empty_after - 8) * 0.5  # Neutral at 8 empty cells
    shaped_rewards += empty_bonus * grid_changed
    
    # 2. Max tile progress bonus
    # Big bonus when reaching new max tiles
    max_before = np.max(states, axis=(1, 2))
    max_after = np.max(next_states, axis=(1, 2))
    new_max_mask = max_after > max_before
    # Exponential bonus for new max tiles: 2^tile_value
    max_tile_bonus = np.where(new_max_mask, 2 ** max_after, 0).astype(np.float32)
    shaped_rewards += max_tile_bonus
    
    # 3. Corner bonus (encourage keeping max tile in corner)
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    corner_values = np.array([[next_states[:, r, c] for r, c in corners]]).T.squeeze()
    max_vals = np.max(next_states, axis=(1, 2))
    corner_has_max = np.any(corner_values == max_vals[:, None], axis=1)
    corner_bonus = np.where(corner_has_max & grid_changed, 2.0, 0.0)
    shaped_rewards += corner_bonus
    
    # 4. Monotonicity bonus (reward for tiles decreasing from corner)
    # Check if tiles are roughly monotonic along rows/cols from top-left
    mono_bonus = np.zeros(num_envs, dtype=np.float32)
    for i in range(num_envs):
        if not grid_changed[i]:
            continue
        grid = next_states[i]
        # Check monotonicity along rows (left to right should decrease or stay same)
        row_mono = 0
        for row in range(4):
            for col in range(3):
                if grid[row, col] >= grid[row, col + 1]:
                    row_mono += 0.1
        # Check monotonicity along columns (top to bottom should decrease or stay same)  
        col_mono = 0
        for col in range(4):
            for row in range(3):
                if grid[row, col] >= grid[row + 1, col]:
                    col_mono += 0.1
        mono_bonus[i] = max(row_mono, col_mono)
    shaped_rewards += mono_bonus
    
    # 5. Smoothness penalty (penalize large differences between adjacent tiles)
    # This encourages merges by keeping similar values together
    smoothness = np.zeros(num_envs, dtype=np.float32)
    for i in range(num_envs):
        if not grid_changed[i]:
            continue
        grid = next_states[i]
        for row in range(4):
            for col in range(3):
                if grid[row, col] > 0 and grid[row, col + 1] > 0:
                    diff = abs(int(grid[row, col]) - int(grid[row, col + 1]))
                    smoothness[i] -= diff * 0.1
        for col in range(4):
            for row in range(3):
                if grid[row, col] > 0 and grid[row + 1, col] > 0:
                    diff = abs(int(grid[row, col]) - int(grid[row + 1, col]))
                    smoothness[i] -= diff * 0.1
    shaped_rewards += smoothness
    
    return shaped_rewards


class ImprovedVectorizedDQNTrainer:
    """
    Improved DQN Trainer with:
    - Better network architecture (V2)
    - Reward shaping
    - Faster epsilon schedule
    - Better hyperparameters
    """
    
    def __init__(
        self,
        num_envs: int = 32,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,  # Steps to decay from start to end
        batch_size: int = 256,
        buffer_capacity: int = 500000,
        target_update_freq: int = 1000,
        invalid_move_penalty: float = -5.0,  # Reduced from -10
        use_double_dqn: bool = True,
        use_reward_shaping: bool = True,
        train_steps_per_collect: int = 2,
        use_large_network: bool = False,
        device: str = None,
        seed: Optional[int] = None
    ):
        self.num_envs = num_envs
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        # Linear decay: eps = eps_start - step * (eps_start - eps_end) / decay_steps
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.invalid_move_penalty = invalid_move_penalty
        self.use_double_dqn = use_double_dqn
        self.use_reward_shaping = use_reward_shaping
        self.train_steps_per_collect = train_steps_per_collect
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        log(f"Using device: {self.device}")
        log(f"Number of parallel environments: {num_envs}")
        
        # Create environment
        self.env = VectorizedGame2048(num_envs=num_envs, seed=seed)
        
        # Initialize networks (V2 with better architecture)
        NetworkClass = DQN2048_V2_Large if use_large_network else DQN2048_V2
        self.policy_net = NetworkClass().to(self.device)
        self.target_net = NetworkClass().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        # Mixed precision
        self.scaler = GradScaler(enabled=str(self.device) != 'cpu')
        self.use_mixed_precision = str(self.device) != 'cpu'
        
        # Replay buffer
        self.replay_buffer = VectorizedReplayBuffer(
            capacity=buffer_capacity,
            state_shape=(4, 4)
        )
        
        # Statistics
        self.total_steps = 0
        self.total_transitions = 0
        self.episodes_completed = 0
        self.training_updates = 0
        
        self.training_history = {
            'step_rewards': [],
            'step_scores': [],
            'max_tiles': [],
            'losses': [],
            'epsilons': [],
            'episodes_completed': [],
            'transitions_per_sec': [],
            'buffer_size': [],
            'eval_scores': [],
            'eval_max_tiles': []
        }
        
        self.recent_scores = []
        self.recent_max_tiles = []
        self.best_max_tile = 0
        self.best_eval_score = 0.0  # Track best eval score for _best.pth saving
        self.eval_score_ema = 0.0   # Exponential moving average for stable best model selection
        self.ema_alpha = 0.3        # EMA smoothing factor (0.3 = recent scores weighted more)
    
    def select_actions(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        """Select actions using epsilon-greedy."""
        if training:
            random_mask = np.random.random(self.num_envs) < self.epsilon
            
            with torch.no_grad():
                states_tensor = torch.from_numpy(states).float().to(self.device)
                self.policy_net.eval()
                q_values = self.policy_net(states_tensor)
                self.policy_net.train()
                greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            
            random_actions = np.random.randint(0, 4, size=self.num_envs)
            return np.where(random_mask, random_actions, greedy_actions)
        else:
            with torch.no_grad():
                states_tensor = torch.from_numpy(states).float().to(self.device)
                self.policy_net.eval()
                q_values = self.policy_net(states_tensor)
                return q_values.argmax(dim=1).cpu().numpy()
    
    def train_step(self) -> float:
        """One gradient update step."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, device=self.device
        )
        
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(dim=1)[0]
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        with autocast(device_type=str(self.device).split(':')[0], enabled=self.use_mixed_precision):
            current_q_values = self.policy_net(states).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)
            loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.training_updates += 1
        return loss.item()
    
    def update_epsilon(self):
        """Update epsilon with linear decay."""
        if self.total_steps < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - self.total_steps * self.epsilon_decay_rate
        else:
            self.epsilon = self.epsilon_end
    
    def collect_and_train(self, num_steps: int) -> Dict[str, float]:
        """Collect experience and train."""
        states = self.env.grids.copy()
        
        total_reward = 0
        total_loss = 0
        num_losses = 0
        completed_episodes = 0
        
        start_time = time.time()
        
        for _ in range(num_steps):
            actions = self.select_actions(states, training=True)
            next_states, rewards, dones, info = self.env.step(actions)
            
            # Apply reward shaping if enabled
            if self.use_reward_shaping:
                rewards = compute_reward_shaping(
                    states, next_states, rewards, dones, info['grid_changed']
                )
            
            # Apply invalid move penalty
            invalid_mask = ~info['grid_changed']
            rewards[invalid_mask] += self.invalid_move_penalty
            
            # Store in buffer
            self.replay_buffer.push_batch(
                states, actions, rewards, next_states, dones.astype(np.float32)
            )
            
            total_reward += rewards.sum()
            completed_episodes += dones.sum()
            
            # Track episode scores
            if np.any(dones) and len(info['final_scores']) > 0:
                for score in info['final_scores']:
                    self.recent_scores.append(score)
                for max_tile in info['final_max_tiles']:
                    tile_val = 2 ** max_tile if max_tile > 0 else 0
                    self.recent_max_tiles.append(tile_val)
                    if tile_val > self.best_max_tile:
                        self.best_max_tile = tile_val
                        if tile_val >= 512:
                            log(f"\n🎉 NEW BEST TILE: {tile_val}!")
            
            # Train if buffer is sufficiently full
            if len(self.replay_buffer) >= self.batch_size * 10:
                for _ in range(self.train_steps_per_collect):
                    loss = self.train_step()
                    if loss > 0:
                        total_loss += loss
                        num_losses += 1
            
            # Update target network
            if self.training_updates % self.target_update_freq == 0 and self.training_updates > 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Update epsilon (linear decay)
            self.update_epsilon()
            
            states = self.env.grids.copy()
            self.total_steps += 1
            self.total_transitions += self.num_envs
        
        elapsed = time.time() - start_time
        transitions_per_sec = (num_steps * self.num_envs) / elapsed
        self.episodes_completed += completed_episodes
        
        return {
            'avg_reward': total_reward / (num_steps * self.num_envs),
            'avg_loss': total_loss / num_losses if num_losses > 0 else 0,
            'completed_episodes': completed_episodes,
            'transitions_per_sec': transitions_per_sec,
            'elapsed_time': elapsed
        }
    
    def evaluate(self, num_episodes: int = 50) -> Dict[str, float]:
        """Evaluate without exploration. Uses 50 episodes by default for stability."""
        eval_env = VectorizedGame2048(num_envs=1, seed=99)
        
        scores = []
        max_tiles = []
        
        for _ in range(num_episodes):
            state = eval_env.reset()[0:1]
            done = False
            
            while not done:
                action = self.select_actions(state, training=False)
                next_state, _, dones, info = eval_env.step(action)
                done = dones[0]
                state = eval_env.grids.copy()
            
            scores.append(info['scores'][0])
            max_tile = 2 ** info['max_tiles'][0] if info['max_tiles'][0] > 0 else 0
            max_tiles.append(max_tile)
        
        return {
            'avg_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': int(np.max(scores)),
            'median_max_tile': int(np.median(max_tiles)),
            'max_tile': int(np.max(max_tiles)),
            'pct_512': 100 * np.mean(np.array(max_tiles) >= 512),
            'pct_1024': 100 * np.mean(np.array(max_tiles) >= 1024),
            'pct_2048': 100 * np.mean(np.array(max_tiles) >= 2048)
        }
    
    def train(
        self,
        total_steps: int,
        log_freq: int = 100,
        eval_freq: int = 1000,
        save_freq: int = 5000,
        save_path: str = "models/dqn_2048_v2.pth"
    ):
        """Main training loop."""
        start_step = self.total_steps
        target_steps = start_step + total_steps
        
        log(f"\n{'='*70}")
        log(f"Starting Improved Vectorized DQN Training (V2)")
        log(f"{'='*70}")
        if start_step > 0:
            log(f"Resuming from step {start_step}")
        log(f"Target steps: {target_steps}")
        log(f"Parallel environments: {self.num_envs}")
        log(f"Network parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")
        log(f"Device: {self.device}")
        log(f"\nKey Improvements:")
        log(f"  - One-hot encoding + CNN architecture")
        log(f"  - Dueling DQN")
        log(f"  - Reward shaping: {self.use_reward_shaping}")
        log(f"  - Linear epsilon decay over {self.epsilon_decay_steps} steps")
        log(f"  - Current epsilon: {self.epsilon:.4f}")
        log(f"  - Invalid move penalty: {self.invalid_move_penalty}")
        log(f"{'='*70}\n")
        
        self.env.reset()
        
        pbar = tqdm(total=total_steps, desc="Training")
        
        steps_done = 0
        while steps_done < total_steps:
            batch_steps = min(log_freq, total_steps - steps_done)
            stats = self.collect_and_train(batch_steps)
            steps_done += batch_steps
            pbar.update(batch_steps)
            
            # Record history
            self.training_history['step_rewards'].append(stats['avg_reward'])
            self.training_history['losses'].append(stats['avg_loss'])
            self.training_history['epsilons'].append(self.epsilon)
            self.training_history['transitions_per_sec'].append(stats['transitions_per_sec'])
            self.training_history['buffer_size'].append(len(self.replay_buffer))
            self.training_history['episodes_completed'].append(self.episodes_completed)
            
            if self.recent_scores:
                self.training_history['step_scores'].append(np.mean(self.recent_scores[-100:]))
                self.training_history['max_tiles'].append(max(self.recent_max_tiles[-100:]))
            
            current_step = self.total_steps
            
            # Logging
            if current_step % eval_freq == 0:
                log(f"\n[Step {current_step}/{target_steps}]")
                log(f"  Transitions: {self.total_transitions:,} | Episodes: {self.episodes_completed:,}")
                log(f"  Epsilon: {self.epsilon:.4f}")
                log(f"  Buffer: {len(self.replay_buffer):,}")
                
                if self.recent_scores:
                    avg_score = np.mean(self.recent_scores[-100:])
                    max_tile = max(self.recent_max_tiles[-100:])
                    log(f"  Train Avg Score (100 ep): {avg_score:.1f}")
                    log(f"  Train Best Tile (100 ep): {max_tile}")
                    log(f"  Best Tile Ever: {self.best_max_tile}")
                
                # Evaluate with more episodes for stability
                eval_stats = self.evaluate(num_episodes=50)
                self.training_history['eval_scores'].append(eval_stats['avg_score'])
                self.training_history['eval_max_tiles'].append(eval_stats['median_max_tile'])
                
                # Update EMA of eval score for stable best model selection
                if self.eval_score_ema == 0:
                    self.eval_score_ema = eval_stats['avg_score']
                else:
                    self.eval_score_ema = (self.ema_alpha * eval_stats['avg_score'] + 
                                           (1 - self.ema_alpha) * self.eval_score_ema)
                
                log(f"  Eval Score: {eval_stats['avg_score']:.1f} ± {eval_stats['std_score']:.1f} (EMA: {self.eval_score_ema:.1f})")
                log(f"  Eval Max Tile (median): {eval_stats['median_max_tile']}")
                log(f"  Eval Max Tile (best): {eval_stats['max_tile']}")
                if eval_stats['pct_512'] > 0:
                    log(f"  512+ rate: {eval_stats['pct_512']:.1f}%")
                if eval_stats['pct_1024'] > 0:
                    log(f"  1024+ rate: {eval_stats['pct_1024']:.1f}%")
                if eval_stats['pct_2048'] > 0:
                    log(f"  🎉 2048+ rate: {eval_stats['pct_2048']:.1f}%")
                
                # Use EMA score for best model selection (more stable than single eval)
                if self.eval_score_ema > self.best_eval_score:
                    self.best_eval_score = self.eval_score_ema
                    self.save_model(save_path.replace('.pth', '_best.pth'))
                    log(f"  >> New best EMA score: {self.best_eval_score:.1f}")
            
            # Save checkpoint
            if current_step % save_freq == 0 and current_step > start_step:
                self.save_model(save_path.replace('.pth', f'_step{current_step}.pth'))
        
        pbar.close()
        log(f"\n{'='*70}")
        log("Training Complete!")
        log(f"{'='*70}")
        log(f"Total steps: {self.total_steps}")
        log(f"Total transitions: {self.total_transitions:,}")
        log(f"Best tile ever: {self.best_max_tile}")
        log(f"Best eval score: {self.best_eval_score:.1f}")
        
        self.save_model(save_path)
        self.plot_training_curves()
    
    def save_model(self, path: str):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_transitions': self.total_transitions,
            'episodes_completed': self.episodes_completed,
            'training_updates': self.training_updates,
            'epsilon': self.epsilon,
            'best_max_tile': self.best_max_tile,
            'best_eval_score': self.best_eval_score,
            'eval_score_ema': self.eval_score_ema,
            'training_history': self.training_history,
            'recent_scores': self.recent_scores[-1000:],
            'recent_max_tiles': self.recent_max_tiles[-1000:],
            'hyperparameters': {
                'num_envs': self.num_envs,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay_steps': self.epsilon_decay_steps,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'invalid_move_penalty': self.invalid_move_penalty,
                'use_double_dqn': self.use_double_dqn,
                'use_reward_shaping': self.use_reward_shaping
            }
        }
        
        torch.save(checkpoint, path)
        log(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_transitions = checkpoint.get('total_transitions', 0)
        self.episodes_completed = checkpoint.get('episodes_completed', 0)
        self.training_updates = checkpoint.get('training_updates', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.best_max_tile = checkpoint.get('best_max_tile', 0)
        self.best_eval_score = checkpoint.get('best_eval_score', 0.0)
        self.eval_score_ema = checkpoint.get('eval_score_ema', 0.0)
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        if 'recent_scores' in checkpoint:
            self.recent_scores = checkpoint['recent_scores']
        if 'recent_max_tiles' in checkpoint:
            self.recent_max_tiles = checkpoint['recent_max_tiles']
        
        log(f"Loaded checkpoint from {path}")
        log(f"  Steps: {self.total_steps}, Episodes: {self.episodes_completed}")
        log(f"  Epsilon: {self.epsilon:.4f}, Best tile: {self.best_max_tile}")
        log(f"  Best eval score: {self.best_eval_score:.1f}, EMA: {self.eval_score_ema:.1f}")
    
    def plot_training_curves(self, save_path: str = "plots/training_curves_v2.png"):
        """Plot training curves."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        def smooth(data, window=50):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Scores
        if self.training_history['step_scores']:
            axes[0, 0].plot(smooth(self.training_history['step_scores']), label='Train')
            if self.training_history['eval_scores']:
                eval_x = np.linspace(0, len(self.training_history['step_scores']), len(self.training_history['eval_scores']))
                axes[0, 0].plot(eval_x, self.training_history['eval_scores'], 'r-', label='Eval')
            axes[0, 0].set_title('Average Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Max tiles
        if self.training_history['max_tiles']:
            axes[0, 1].plot(self.training_history['max_tiles'])
            axes[0, 1].axhline(y=2048, color='g', linestyle='--', label='Goal: 2048')
            axes[0, 1].set_title('Max Tile')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Losses
        losses = [l for l in self.training_history['losses'] if l > 0]
        if losses:
            axes[0, 2].plot(smooth(losses))
            axes[0, 2].set_title('Training Loss')
            axes[0, 2].grid(True)
        
        # Epsilon
        if self.training_history['epsilons']:
            axes[1, 0].plot(self.training_history['epsilons'])
            axes[1, 0].set_title('Epsilon')
            axes[1, 0].grid(True)
        
        # Rewards
        if self.training_history['step_rewards']:
            axes[1, 1].plot(smooth(self.training_history['step_rewards']))
            axes[1, 1].set_title('Average Reward')
            axes[1, 1].grid(True)
        
        # Throughput
        if self.training_history['transitions_per_sec']:
            axes[1, 2].plot(smooth(self.training_history['transitions_per_sec']))
            axes[1, 2].set_title('Transitions/sec')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        log(f"Training curves saved to {save_path}")


def main(
    resume_from: str = None,
    num_envs: int = 32,
    total_steps: int = 200000,
    batch_size: int = 256,
    use_large_network: bool = False,
    learning_rate: float = None  # None means use default (1e-4) or checkpoint value
):
    """Main training function."""
    global logger
    logger = setup_logging()
    
    # Use default LR for fresh training, can be overridden after loading checkpoint
    initial_lr = learning_rate if learning_rate else 1e-4
    
    trainer = ImprovedVectorizedDQNTrainer(
        num_envs=num_envs,
        learning_rate=initial_lr,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,  # Slightly higher minimum for continued exploration
        epsilon_decay_steps=100000,  # Linear decay over 100k steps (vs 200k+ before)
        batch_size=batch_size,
        buffer_capacity=500000,
        target_update_freq=1000,
        invalid_move_penalty=-5.0,  # Reduced from -10
        use_double_dqn=True,
        use_reward_shaping=True,
        train_steps_per_collect=2,
        use_large_network=use_large_network,
        seed=42
    )
    
    if resume_from:
        log(f"\nResuming from: {resume_from}")
        trainer.load_model(resume_from)
        
        # Override learning rate if specified
        if learning_rate:
            old_lr = trainer.optimizer.param_groups[0]['lr']
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = learning_rate
            log(f"Learning rate overridden: {old_lr} -> {learning_rate}")
    
    trainer.train(
        total_steps=total_steps,
        log_freq=100,
        eval_freq=1000,
        save_freq=5000,
        save_path="models/dqn_2048_v2.pth"
    )
    
    # Final evaluation
    log(f"\n{'='*70}")
    log("Final Evaluation (50 episodes)")
    log(f"{'='*70}")
    final_stats = trainer.evaluate(num_episodes=50)
    log(f"Average Score: {final_stats['avg_score']:.1f} ± {final_stats['std_score']:.1f}")
    log(f"Max Score: {final_stats['max_score']}")
    log(f"Median Max Tile: {final_stats['median_max_tile']}")
    log(f"Best Max Tile: {final_stats['max_tile']}")
    log(f"512+ rate: {final_stats['pct_512']:.1f}%")
    log(f"1024+ rate: {final_stats['pct_1024']:.1f}%")
    log(f"2048+ rate: {final_stats['pct_2048']:.1f}%")
    
    with open('models/final_stats_v2.json', 'w') as f:
        json.dump(final_stats, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved 2048 DQN Training V2')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num-envs', type=int, default=32,
                        help='Number of parallel environments')
    parser.add_argument('--steps', type=int, default=200000,
                        help='Total environment steps')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-4, use lower like 5e-5 for fine-tuning)')
    parser.add_argument('--large', action='store_true',
                        help='Use larger network')
    
    args = parser.parse_args()
    
    main(
        resume_from=args.resume,
        num_envs=args.num_envs,
        total_steps=args.steps,
        batch_size=args.batch_size,
        use_large_network=args.large,
        learning_rate=args.lr
    )
