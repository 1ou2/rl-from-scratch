"""
Training script for 2048 game using Deep Q-Learning (DQN)
This implements a complete DQN training loop with experience replay and target networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

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
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions randomly.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
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
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100000,
        target_update_freq: int = 1000,
        invalid_move_penalty: float = -10.0,
        use_double_dqn: bool = True,
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
            'epsilons': []
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
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def update_target_network(self) -> None:
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train_step(self) -> float:
        """
        Perform one training step (sample batch and update network).
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values: Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
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
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()
    
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
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select and perform action
            action = self.select_action(state, training=True)
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Apply penalty for invalid moves
            if not info['grid_changed']:
                reward += self.invalid_move_penalty
            
            # Store transition in replay buffer
            self.replay_buffer.push(
                state, action, reward, next_state, 
                float(done)  # Only terminal states, not truncated
            )
            
            # Perform training step
            loss = self.train_step()
            if loss > 0:
                episode_loss += loss
                num_updates += 1
            
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
            'buffer_size': len(self.replay_buffer)
        }
        
        self.episodes += 1
        return stats
    
    def train(self, num_episodes: int, eval_freq: int = 100, save_freq: int = 500,
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
        print(f"Hyperparameters:")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Gamma: {self.gamma}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Epsilon: {self.epsilon_start} -> {self.epsilon_end} (decay: {self.epsilon_decay})")
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
            
            # Update best score
            if stats['score'] > best_score:
                best_score = stats['score']
            
            # Use self.episodes (cumulative) instead of episode (loop counter)
            current_episode = self.episodes
            
            # Periodic evaluation and logging
            if current_episode % eval_freq == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-eval_freq:])
                avg_score = np.mean(self.training_history['episode_scores'][-eval_freq:])
                recent_tiles = self.training_history['max_tiles'][-eval_freq:]
                median_max_tile = np.median(recent_tiles)
                max_tile_achieved = np.max(recent_tiles)
                avg_length = np.mean(self.training_history['episode_lengths'][-eval_freq:])
                
                print(f"\nEpisode {current_episode}/{target_episodes}")
                print(f"  Avg Reward (last {eval_freq}): {avg_reward:.2f}")
                print(f"  Avg Score (last {eval_freq}): {avg_score:.2f}")
                print(f"  Median Max Tile (last {eval_freq}): {median_max_tile:.0f}")
                print(f"  Best Max Tile (last {eval_freq}): {max_tile_achieved:.0f}")
                print(f"  Avg Length (last {eval_freq}): {avg_length:.1f}")
                print(f"  Best Score: {best_score}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                print(f"  Buffer size: {len(self.replay_buffer)}")
                
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
            'training_history': self.training_history
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
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=100000,
        target_update_freq=1000,
        invalid_move_penalty=-10.0,
        use_double_dqn=True
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
        num_episodes=5000,
        eval_freq=100,
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
