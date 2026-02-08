"""
Training script for 2048 game using Deep Q-Learning (DQN)
This implements a complete DQN training loop with experience replay and target networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler  # Mixed precision training
from collections import deque
import random
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import time

from game_2048 import Game2048
from neural_network import DQN2048


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    
    Stores transitions (state, action, reward, next_state, done) and enables
    random sampling to break temporal correlations in the training data.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int, device: torch.device = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions randomly.
        
        Args:
            batch_size: Number of samples to retrieve
            device: Device to place tensors on (GPU or CPU)
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Stack numpy arrays without intermediate conversion
        states_array = np.stack(states, axis=0)
        next_states_array = np.stack(next_states, axis=0)
        
        # Create tensors directly on target device (avoids CPU overhead)
        if device is not None:
            return (
                torch.from_numpy(states_array).float().to(device),
                torch.tensor(actions, dtype=torch.long, device=device),
                torch.tensor(rewards, dtype=torch.float32, device=device),
                torch.from_numpy(next_states_array).float().to(device),
                torch.tensor(dones, dtype=torch.float32, device=device)
            )
        else:
            return (
                torch.FloatTensor(states_array),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states_array),
                torch.FloatTensor(dones)
            )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNTrainer:
    """
    DQN Trainer for 2048 Game.
    
    Implements Deep Q-Learning with:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration with decay
    - Invalid move penalties
    - Double DQN (optional)
    """
    
    def __init__(
        self,
        env: Game2048,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,  # Much slower decay (0.9995 vs 0.995) to maintain exploration longer
        batch_size: int = 256,  # Increased for better GPU utilization
        buffer_capacity: int = 200000,  # Increased for better experience diversity
        target_update_freq: int = 1000,
        invalid_move_penalty: float = -10.0,
        use_double_dqn: bool = True,
        train_steps_per_episode: int = 4,  # Multiple training passes per game step
        device: str = None
    ):
        """
        Initialize DQN trainer.
        
        Args:
            env: Game2048 environment
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            batch_size: Batch size for training
            buffer_capacity: Replay buffer capacity
            target_update_freq: Steps between target network updates
            invalid_move_penalty: Penalty for invalid moves
            use_double_dqn: Whether to use Double DQN
            train_steps_per_episode: Number of training steps per game step (GPU utilization)
            device: Device to use ('cuda' or 'cpu')
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.invalid_move_penalty = invalid_move_penalty
        self.use_double_dqn = use_double_dqn
        self.train_steps_per_episode = train_steps_per_episode
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = DQN2048().to(self.device)
        self.target_net = DQN2048().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss, more robust than MSE
        
        # Mixed precision training (faster on GPU)
        self.scaler = GradScaler(enabled=str(self.device) != 'cpu')
        self.use_mixed_precision = str(self.device) != 'cpu'
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.steps = 0
        self.episodes = 0
        self.training_history = {
            'episode_rewards': [],
            'episode_scores': [],
            'episode_lengths': [],
            'max_tiles': [],
            'losses': [],
            'epsilons': [],
            'timing_game_steps': [],
            'timing_train_sampling': [],
            'timing_train_forward': [],
            'timing_train_backward': []
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor)
            self.policy_net.train()
            return q_values.argmax(dim=1).item()
    
    def update_target_network(self) -> None:
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train_step(self, num_steps: int = 1) -> Tuple[float, Dict[str, float]]:
        """
        Perform multiple training steps (sample batch and update network).
        
        Args:
            num_steps: Number of training steps to perform in this call
        
        Returns:
            Tuple of (average_loss, timings_dict)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, {}
        
        total_loss = 0.0
        timings = {'sampling': 0.0, 'forward': 0.0, 'backward': 0.0}
        
        for _ in range(num_steps):
            # Sample batch from replay buffer (optimized - tensors directly on device)
            t0 = time.time()
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size, device=self.device
            )
            timings['sampling'] += time.time() - t0
            
            # Compute target Q-values
            with torch.no_grad():
                if self.use_double_dqn:
                    # Double DQN: use policy net to select action, target net to evaluate
                    next_actions = self.policy_net(next_states).argmax(dim=1)
                    next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    # Standard DQN
                    next_q_values = self.target_net(next_states).max(dim=1)[0]
                
                # Target: r + gamma * max_a Q_target(s', a) * (1 - done)
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            # Mixed precision forward pass (faster on GPU)
            t0 = time.time()
            with autocast(device_type=str(self.device).split(':')[0], enabled=self.use_mixed_precision):
                # Current Q-values: Q(s, a)
                current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                # Compute loss
                loss = self.criterion(current_q_values, target_q_values)
            timings['forward'] += time.time() - t0
            
            # Optimize the model with gradient scaling
            t0 = time.time()
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            timings['backward'] += time.time() - t0
            
            total_loss += loss.item()
        
        # Average timings
        for key in timings:
            timings[key] /= num_steps
        
        return total_loss / num_steps, timings
    
    def train_episode(self) -> Dict[str, Any]:
        """
        Train for one complete episode.
        
        Returns:
            Episode statistics
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        num_updates = 0
        
        # Timing breakdown
        timing_game_steps = 0.0
        timing_train_sampling = 0.0
        timing_train_forward = 0.0
        timing_train_backward = 0.0
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Time the game step
            t0 = time.time()
            action = self.select_action(state, training=True)
            next_state, reward, done, truncated, info = self.env.step(action)
            timing_game_steps += time.time() - t0
            
            # Apply penalty for invalid moves
            if not info['grid_changed']:
                reward += self.invalid_move_penalty
            
            # Store transition in replay buffer
            self.replay_buffer.push(
                state, action, reward, next_state, 
                float(done)  # Only terminal states, not truncated
            )
            
            # Perform multiple training steps per game step (GPU utilization)
            loss, timings = self.train_step(num_steps=self.train_steps_per_episode)
            if loss > 0:
                episode_loss += loss
                num_updates += 1
                timing_train_sampling += timings['sampling']
                timing_train_forward += timings['forward']
                timing_train_backward += timings['backward']
            
            # Update target network periodically
            if self.steps % self.target_update_freq == 0:
                self.update_target_network()
            
            state = next_state
            episode_reward += reward
            self.steps += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Episode statistics
        max_tile = 2 ** state.max() if state.max() > 0 else 0
        avg_loss = episode_loss / num_updates if num_updates > 0 else 0
        
        stats = {
            'reward': episode_reward,
            'score': info['score'],
            'length': info['total_steps'],
            'max_tile': max_tile,
            'loss': avg_loss,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            # Timing breakdown
            'timing_game_steps': timing_game_steps,
            'timing_train_sampling': timing_train_sampling,
            'timing_train_forward': timing_train_forward,
            'timing_train_backward': timing_train_backward
        }
        
        self.episodes += 1
        return stats
    
    def train(self, num_episodes: int, eval_freq: int = 200, save_freq: int = 500,
              save_path: str = "models/dqn_2048.pth") -> None:
        """
        Train the agent for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to train
            eval_freq: Frequency of evaluation episodes
            save_freq: Frequency of model saves
            save_path: Path to save model checkpoints
        """
        start_episode = self.episodes  # Track where we started (for resumed training)
        target_episodes = start_episode + num_episodes
        
        print(f"\nStarting training for {num_episodes} episodes...")
        if start_episode > 0:
            print(f"Resuming from episode {start_episode}, training until episode {target_episodes}")
        print(f"Policy Network: {sum(p.numel() for p in self.policy_net.parameters())} parameters")
        print(f"Device: {self.device} | Mixed Precision: {self.use_mixed_precision}")
        print(f"Hyperparameters:")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Gamma: {self.gamma}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Train steps per episode: {self.train_steps_per_episode}")
        print(f"  - Epsilon: {self.epsilon_start} -> {self.epsilon_end} (decay: {self.epsilon_decay})")
        print(f"  - Epsilon schedule: decay 0.9995/ep (~15% at ep15k), reset to 0.15 at ep30k")
        print(f"  - Current epsilon: {self.epsilon:.4f}")
        print(f"  - Buffer capacity: {self.replay_buffer.buffer.maxlen}")
        print(f"  - Target update freq: {self.target_update_freq}")
        print(f"  - Double DQN: {self.use_double_dqn}")
        print(f"  - Invalid move penalty: {self.invalid_move_penalty}")
        print()
        
        best_score = 0
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            stats = self.train_episode()
            
            # Record statistics
            self.training_history['episode_rewards'].append(stats['reward'])
            self.training_history['episode_scores'].append(stats['score'])
            self.training_history['episode_lengths'].append(stats['length'])
            self.training_history['max_tiles'].append(stats['max_tile'])
            self.training_history['losses'].append(stats['loss'])
            self.training_history['epsilons'].append(stats['epsilon'])
            self.training_history['timing_game_steps'].append(stats['timing_game_steps'])
            self.training_history['timing_train_sampling'].append(stats['timing_train_sampling'])
            self.training_history['timing_train_forward'].append(stats['timing_train_forward'])
            self.training_history['timing_train_backward'].append(stats['timing_train_backward'])
            
            # Update best score
            if stats['score'] > best_score:
                best_score = stats['score']
            
            # Use self.episodes (cumulative) instead of episode (loop counter)
            current_episode = self.episodes
            
            # Reset epsilon for renewed exploration at milestone (helps break plateaus)
            if current_episode == 30000 and self.epsilon == self.epsilon_end:
                old_epsilon = self.epsilon
                self.epsilon = 0.15  # Higher reset: 15% exploration to discover new tiles
                print(f"\n🔄 Epsilon reset at episode {current_episode}: {old_epsilon:.4f} → {self.epsilon:.4f} (15% exploration for new discoveries)\n")
            
            # Periodic evaluation and logging
            if current_episode % eval_freq == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-eval_freq:])
                avg_score = np.mean(self.training_history['episode_scores'][-eval_freq:])
                recent_tiles = self.training_history['max_tiles'][-eval_freq:]
                median_max_tile = np.median(recent_tiles)
                max_tile_achieved = np.max(recent_tiles)
                avg_length = np.mean(self.training_history['episode_lengths'][-eval_freq:])
                
                # Timing breakdown
                avg_game_time = np.mean(self.training_history['timing_game_steps'][-eval_freq:])
                avg_sampling_time = np.mean(self.training_history['timing_train_sampling'][-eval_freq:])
                avg_forward_time = np.mean(self.training_history['timing_train_forward'][-eval_freq:])
                avg_backward_time = np.mean(self.training_history['timing_train_backward'][-eval_freq:])
                total_train_time = avg_sampling_time + avg_forward_time + avg_backward_time
                
                print(f"\nEpisode {current_episode}/{target_episodes}")
                print(f"  Avg Reward (last {eval_freq}): {avg_reward:.2f}")
                print(f"  Avg Score (last {eval_freq}): {avg_score:.2f}")
                print(f"  Median Max Tile (last {eval_freq}): {median_max_tile:.0f}")
                print(f"  Best Max Tile (last {eval_freq}): {max_tile_achieved:.0f}")
                print(f"  Avg Length (last {eval_freq}): {avg_length:.1f}")
                print(f"  Best Score: {best_score}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                print(f"  Buffer size: {len(self.replay_buffer)}")
                print(f"\n  ⏱️  Timing Breakdown (per episode):")
                print(f"    - Game steps: {avg_game_time*1000:.1f}ms ({avg_game_time/(avg_game_time+total_train_time)*100:.1f}%)")
                print(f"    - Training: {total_train_time*1000:.1f}ms ({total_train_time/(avg_game_time+total_train_time)*100:.1f}%)")
                print(f"      └─ Sampling: {avg_sampling_time*1000:.2f}ms")
                print(f"      └─ Forward: {avg_forward_time*1000:.2f}ms")
                print(f"      └─ Backward: {avg_backward_time*1000:.2f}ms")
                
                # Evaluate agent
                eval_stats = self.evaluate(num_episodes=5)
                print(f"  Eval Score: {eval_stats['avg_score']:.2f} ± {eval_stats['std_score']:.2f}")
                print(f"  Eval Max Tile (median): {eval_stats['median_max_tile']:.0f}, (best): {eval_stats['max_tile']:.0f}")
            
            # Save model periodically using actual episode count
            if current_episode % save_freq == 0 and current_episode > start_episode:
                self.save_model(save_path.replace('.pth', f'_ep{current_episode}.pth'))
        
        print("\nTraining complete!")
        print(f"Final best score: {best_score}")
        
        # Save final model with actual episode count
        final_save_path = save_path.replace('.pth', f'_ep{self.episodes}.pth')
        self.save_model(final_save_path)
        print(f"Final model also saved as: {save_path}")
        self.save_model(save_path)  # Save as default name too
        
        # Plot training curves
        self.plot_training_curves()
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent without exploration.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation statistics
        """
        scores = []
        max_tiles = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.select_action(state, training=False)
                state, _, done, truncated, info = self.env.step(action)
            
            scores.append(info['score'])
            max_tile = 2 ** state.max() if state.max() > 0 else 0
            max_tiles.append(max_tile)
        
        return {
            'avg_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': int(np.max(scores)),
            'median_max_tile': int(np.median(max_tiles)),
            'max_tile': int(np.max(max_tiles))
        }
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            # Save hyperparameters for validation on resume
            'hyperparameters': {
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'invalid_move_penalty': self.invalid_move_penalty,
                'use_double_dqn': self.use_double_dqn,
                'train_steps_per_episode': self.train_steps_per_episode
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episodes = checkpoint['episodes']
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
        self.training_history = checkpoint['training_history']
        
        # Ensure all required timing keys exist (for backward compatibility with old checkpoints)
        timing_keys = ['timing_game_steps', 'timing_train_sampling', 'timing_train_forward', 'timing_train_backward']
        for key in timing_keys:
            if key not in self.training_history:
                self.training_history[key] = [0.0] * len(self.training_history.get('episode_rewards', []))
        
        # Validate hyperparameters if they were saved in checkpoint
        if 'hyperparameters' in checkpoint:
            saved_hparams = checkpoint['hyperparameters']
            current_hparams = {
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'invalid_move_penalty': self.invalid_move_penalty,
                'use_double_dqn': self.use_double_dqn,
                'train_steps_per_episode': self.train_steps_per_episode
            }
            
            mismatches = []
            for key in saved_hparams:
                if key in current_hparams and saved_hparams[key] != current_hparams[key]:
                    mismatches.append(
                        f"  {key}: saved={saved_hparams[key]}, current={current_hparams[key]}"
                    )
            
            if mismatches:
                print(f"\n{'='*60}")
                print("WARNING: Hyperparameter mismatch detected!")
                print("The following hyperparameters differ from the checkpoint:")
                for msg in mismatches:
                    print(msg)
                print(f"{'='*60}\n")
                print("Consider updating main() to match the saved hyperparameters.")
        
        print(f"Model loaded from {path}")
        print(f"Resumed at episode {self.episodes}, step {self.steps}")
    
    def plot_training_curves(self, save_path: str = "plots/training_curves.png") -> None:
        """Plot training curves."""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Smooth curves using moving average
        window = 100
        
        def moving_average(data, window):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Plot scores
        axes[0, 0].plot(moving_average(self.training_history['episode_scores'], window))
        axes[0, 0].set_title('Episode Score')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True)
        
        # Plot rewards
        axes[0, 1].plot(moving_average(self.training_history['episode_rewards'], window))
        axes[0, 1].set_title('Episode Reward')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)
        
        # Plot max tiles
        axes[0, 2].plot(moving_average(self.training_history['max_tiles'], window))
        axes[0, 2].set_title('Max Tile')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Max Tile Value')
        axes[0, 2].grid(True)
        
        # Plot episode lengths
        axes[1, 0].plot(moving_average(self.training_history['episode_lengths'], window))
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Plot losses
        losses_filtered = [l for l in self.training_history['losses'] if l > 0]
        if len(losses_filtered) > 0:
            axes[1, 1].plot(moving_average(losses_filtered, min(window, len(losses_filtered))))
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        # Plot epsilon
        axes[1, 2].plot(self.training_history['epsilons'])
        axes[1, 2].set_title('Epsilon (Exploration Rate)')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Epsilon')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
        plt.close()


def main(resume_from: str = None):
    """
    Main training function.
    
    Args:
        resume_from: Path to checkpoint to resume from (e.g., 'models/dqn_2048_ep500.pth')
                     If None, starts training from scratch
    """
    # Create environment
    env = Game2048(
        render_mode=None,
        max_episode_steps=2000,
        max_consecutive_invalid_moves=10
    )
    
    # Create trainer
    trainer = DQNTrainer(
        env=env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,  # Slower decay: maintain ~15% exploration even after 15k episodes
        batch_size=256,  # Proven optimal: 384 trades learning for speed (not worth it)
        buffer_capacity=100000,
        target_update_freq=1000,
        invalid_move_penalty=-10.0,
        use_double_dqn=True,
        train_steps_per_episode=2  # Keep balanced: 2 is sweet spot for stability
    )
    
    # Resume from checkpoint if provided
    if resume_from:
        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING FROM CHECKPOINT")
        print(f"{'='*60}")
        trainer.load_model(resume_from)
        print(f"{'='*60}\n")
    
    # Train the agent
    trainer.train(
        num_episodes=30000,
        eval_freq=500,  # Evaluate every 500 episodes for faster training
        save_freq=500,
        save_path="models/dqn_2048.pth"
    )
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation (100 episodes)")
    print("="*50)
    final_stats = trainer.evaluate(num_episodes=100)
    print(f"Average Score: {final_stats['avg_score']:.2f} ± {final_stats['std_score']:.2f}")
    print(f"Max Score: {final_stats['max_score']:.0f}")
    print(f"Median Max Tile: {final_stats['median_max_tile']:.0f}")
    print(f"Best Max Tile: {final_stats['max_tile']:.0f}")
    
    # Save final statistics
    with open('models/final_stats.json', 'w') as f:
        json.dump(final_stats, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train 2048 DQN Agent')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., models/dqn_2048_ep500.pth)')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes to train (default: 5000)')
    
    args = parser.parse_args()
    
    # Update main to accept episodes parameter
    main(resume_from=args.resume)
