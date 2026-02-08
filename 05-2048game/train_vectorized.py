"""
Vectorized Training Script for 2048 Game using Deep Q-Learning (DQN)

This training script uses a vectorized game environment to run multiple games
in parallel, significantly improving training throughput.

Key improvements over single-environment training:
1. N games run in parallel (default: 16)
2. Replay buffer fills N times faster  
3. Better GPU utilization through batched experience collection
4. ~5-10x speedup in wall-clock time

Usage:
    python train_vectorized.py                    # Train from scratch
    python train_vectorized.py --resume models/dqn_2048_ep1000.pth  # Resume
    python train_vectorized.py --num-envs 32      # Use 32 parallel games
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
import os

from vectorized_game_2048 import VectorizedGame2048, VectorizedReplayBuffer
from neural_network import DQN2048


class VectorizedDQNTrainer:
    """
    DQN Trainer optimized for vectorized environments.
    
    Key differences from standard DQNTrainer:
    - Collects N transitions per step (from N parallel games)
    - Uses pre-allocated replay buffer for efficiency
    - Batched action selection across all environments
    - More efficient training loop
    """
    
    def __init__(
        self,
        num_envs: int = 16,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 512,  # Larger batch for better GPU utilization
        buffer_capacity: int = 500000,  # Larger buffer for more diversity
        target_update_freq: int = 1000,
        invalid_move_penalty: float = -10.0,
        use_double_dqn: bool = True,
        train_steps_per_collect: int = 4,  # Training steps per collection step
        device: str = None,
        seed: Optional[int] = None
    ):
        """
        Initialize vectorized DQN trainer.
        
        Args:
            num_envs: Number of parallel environments
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay per step (not per episode!)
            batch_size: Training batch size
            buffer_capacity: Replay buffer capacity
            target_update_freq: Steps between target network updates
            invalid_move_penalty: Penalty for invalid moves
            use_double_dqn: Whether to use Double DQN
            train_steps_per_collect: Training updates per environment step
            device: Device to use ('cuda' or 'cpu')
            seed: Random seed
        """
        self.num_envs = num_envs
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.invalid_move_penalty = invalid_move_penalty
        self.use_double_dqn = use_double_dqn
        self.train_steps_per_collect = train_steps_per_collect
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"Number of parallel environments: {num_envs}")
        
        # Create vectorized environment
        self.env = VectorizedGame2048(num_envs=num_envs, seed=seed)
        
        # Initialize networks
        self.policy_net = DQN2048().to(self.device)
        self.target_net = DQN2048().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=str(self.device) != 'cpu')
        self.use_mixed_precision = str(self.device) != 'cpu'
        
        # Replay buffer (vectorized, pre-allocated)
        self.replay_buffer = VectorizedReplayBuffer(
            capacity=buffer_capacity,
            state_shape=(4, 4)
        )
        
        # Training statistics
        self.total_steps = 0  # Total environment steps (each step = num_envs transitions)
        self.total_transitions = 0  # Total transitions collected
        self.episodes_completed = 0  # Approximate episodes (based on dones)
        self.training_updates = 0  # Number of gradient updates
        
        self.training_history = {
            'step_rewards': [],  # Average reward per step
            'step_scores': [],   # Average score per step
            'max_tiles': [],     # Max tile achieved recently
            'losses': [],
            'epsilons': [],
            'episodes_completed': [],
            'transitions_per_sec': [],
            'buffer_size': []
        }
        
        # For tracking episode statistics
        self.recent_scores = []
        self.recent_max_tiles = []
        self.recent_rewards = []
    
    def select_actions(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select actions for all environments using epsilon-greedy.
        
        Args:
            states: States for all environments, shape (num_envs, 4, 4)
            training: Whether in training mode
        
        Returns:
            Actions for all environments, shape (num_envs,)
        """
        if training:
            # Epsilon-greedy: random actions for some environments
            random_mask = np.random.random(self.num_envs) < self.epsilon
            
            # Get greedy actions from network
            with torch.no_grad():
                states_tensor = torch.from_numpy(states).float().to(self.device)
                self.policy_net.eval()
                q_values = self.policy_net(states_tensor)
                self.policy_net.train()
                greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            
            # Random actions
            random_actions = np.random.randint(0, 4, size=self.num_envs)
            
            # Combine based on epsilon
            actions = np.where(random_mask, random_actions, greedy_actions)
            return actions
        else:
            # Pure greedy for evaluation
            with torch.no_grad():
                states_tensor = torch.from_numpy(states).float().to(self.device)
                self.policy_net.eval()
                q_values = self.policy_net(states_tensor)
                return q_values.argmax(dim=1).cpu().numpy()
    
    def train_step(self) -> float:
        """
        Perform one training step (sample batch and update network).
        
        Returns:
            Loss value (or 0 if buffer too small)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, device=self.device
        )
        
        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(dim=1)[0]
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Forward pass with mixed precision
        with autocast(device_type=str(self.device).split(':')[0], enabled=self.use_mixed_precision):
            current_q_values = self.policy_net(states).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)
            loss = self.criterion(current_q_values, target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.training_updates += 1
        return loss.item()
    
    def update_target_network(self) -> None:
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def collect_and_train(self, num_steps: int) -> Dict[str, float]:
        """
        Collect experience from environments and train.
        
        This is the main training loop step. For each environment step:
        1. Select actions for all N environments
        2. Execute actions and collect N transitions
        3. Store in replay buffer
        4. Perform M training updates

        Args:
            num_steps: Number of environment steps to execute
        
        Returns:
            Statistics dictionary
        """
        states = self.env.grids.copy()
        
        total_reward = 0
        total_loss = 0
        num_losses = 0
        completed_episodes = 0
        
        start_time = time.time()
        
        for _ in range(num_steps):
            # Select actions
            actions = self.select_actions(states, training=True)
            
            # Execute actions (all environments in parallel)
            next_states, rewards, dones, info = self.env.step(actions)
            
            # Apply invalid move penalty
            invalid_mask = ~info['grid_changed']
            rewards[invalid_mask] += self.invalid_move_penalty
            
            # Store transitions in buffer
            self.replay_buffer.push_batch(
                states, actions, rewards, next_states, dones.astype(np.float32)
            )
            
            # Track statistics
            total_reward += rewards.sum()
            completed_episodes += dones.sum()
            
            # Track completed episode scores using FINAL values before reset
            if np.any(dones) and len(info['final_scores']) > 0:
                for score in info['final_scores']:
                    self.recent_scores.append(score)
                for max_tile in info['final_max_tiles']:
                    self.recent_max_tiles.append(2 ** max_tile if max_tile > 0 else 0)
            
            # Perform training updates (but only if buffer has enough samples)
            if len(self.replay_buffer) >= self.batch_size * 10:  # Wait for buffer to fill a bit
                for _ in range(self.train_steps_per_collect):
                    loss = self.train_step()
                    if loss > 0:
                        total_loss += loss
                        num_losses += 1
            
            # Update target network
            if self.training_updates % self.target_update_freq == 0:
                self.update_target_network()
            
            # Decay epsilon (per step, not per episode)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # IMPORTANT: Use env.grids for fresh state after auto-reset
            # next_states contains terminal states (for buffer), but we need reset states for next step
            states = self.env.grids.copy()
            self.total_steps += 1
            self.total_transitions += self.num_envs
        
        elapsed = time.time() - start_time
        transitions_per_sec = (num_steps * self.num_envs) / elapsed
        
        # Update completed episodes count
        self.episodes_completed += completed_episodes
        
        avg_loss = total_loss / num_losses if num_losses > 0 else 0
        avg_reward = total_reward / (num_steps * self.num_envs)
        
        return {
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
            'completed_episodes': completed_episodes,
            'transitions_per_sec': transitions_per_sec,
            'elapsed_time': elapsed
        }
    
    def train(
        self,
        total_steps: int,
        log_freq: int = 100,
        eval_freq: int = 500,
        save_freq: int = 1000,
        save_path: str = "models/dqn_2048_vec.pth"
    ) -> None:
        """
        Train the agent for a specified number of steps.
        
        Args:
            total_steps: Total environment steps (each step generates num_envs transitions)
            log_freq: Steps between logging
            eval_freq: Steps between evaluation
            save_freq: Steps between model saves
            save_path: Path to save checkpoints
        """
        start_step = self.total_steps
        target_steps = start_step + total_steps
        
        print(f"\n{'='*60}")
        print(f"Starting Vectorized DQN Training")
        print(f"{'='*60}")
        if start_step > 0:
            print(f"Resuming from step {start_step}")
        print(f"Target steps: {target_steps}")
        print(f"Parallel environments: {self.num_envs}")
        print(f"Policy Network: {sum(p.numel() for p in self.policy_net.parameters())} parameters")
        print(f"Device: {self.device} | Mixed Precision: {self.use_mixed_precision}")
        print(f"\nHyperparameters:")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Gamma: {self.gamma}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Train steps per collect: {self.train_steps_per_collect}")
        print(f"  - Epsilon: {self.epsilon_start} -> {self.epsilon_end} (decay: {self.epsilon_decay})")
        print(f"  - Current epsilon: {self.epsilon:.4f}")
        print(f"  - Buffer capacity: {self.replay_buffer.capacity}")
        print(f"  - Target update freq: {self.target_update_freq}")
        print(f"  - Double DQN: {self.use_double_dqn}")
        print(f"  - Invalid move penalty: {self.invalid_move_penalty}")
        print(f"{'='*60}\n")
        
        # Reset environments
        self.env.reset()
        
        best_score = 0
        pbar = tqdm(total=total_steps, desc="Training", initial=0)
        
        steps_done = 0
        while steps_done < total_steps:
            # Collect and train for log_freq steps
            batch_steps = min(log_freq, total_steps - steps_done)
            stats = self.collect_and_train(batch_steps)
            steps_done += batch_steps
            pbar.update(batch_steps)
            
            # Record statistics
            self.training_history['step_rewards'].append(stats['avg_reward'])
            self.training_history['losses'].append(stats['avg_loss'])
            self.training_history['epsilons'].append(self.epsilon)
            self.training_history['transitions_per_sec'].append(stats['transitions_per_sec'])
            self.training_history['buffer_size'].append(len(self.replay_buffer))
            self.training_history['episodes_completed'].append(self.episodes_completed)
            
            # Track scores
            if self.recent_scores:
                recent_avg_score = np.mean(self.recent_scores[-100:]) if self.recent_scores else 0
                recent_max_tile = max(self.recent_max_tiles[-100:]) if self.recent_max_tiles else 0
                self.training_history['step_scores'].append(recent_avg_score)
                self.training_history['max_tiles'].append(recent_max_tile)
                
                if recent_avg_score > best_score:
                    best_score = recent_avg_score
            
            current_step = self.total_steps
            
            # Epsilon reset at milestone
            if self.episodes_completed >= 30000 and self.epsilon == self.epsilon_end:
                old_epsilon = self.epsilon
                self.epsilon = 0.15
                print(f"\n🔄 Epsilon reset at {self.episodes_completed} episodes: {old_epsilon:.4f} → {self.epsilon:.4f}")
            
            # Log progress
            if current_step % eval_freq == 0:
                print(f"\n[Step {current_step}/{target_steps}]")
                print(f"  Transitions collected: {self.total_transitions:,}")
                print(f"  Episodes completed: {self.episodes_completed:,}")
                print(f"  Transitions/sec: {stats['transitions_per_sec']:.0f}")
                print(f"  Buffer size: {len(self.replay_buffer):,}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                
                if self.recent_scores:
                    avg_score = np.mean(self.recent_scores[-100:])
                    median_tile = np.median(self.recent_max_tiles[-100:])
                    max_tile = max(self.recent_max_tiles[-100:])
                    print(f"  Avg Score (last 100 ep): {avg_score:.1f}")
                    print(f"  Median Max Tile: {median_tile:.0f}")
                    print(f"  Best Max Tile: {max_tile:.0f}")
                    print(f"  Best Score: {best_score:.1f}")
                
                # Evaluate
                eval_stats = self.evaluate(num_episodes=10)
                print(f"  Eval Score: {eval_stats['avg_score']:.1f} ± {eval_stats['std_score']:.1f}")
                print(f"  Eval Max Tile (median): {eval_stats['median_max_tile']:.0f}")
            
            # Save checkpoint
            if current_step % save_freq == 0 and current_step > start_step:
                checkpoint_path = save_path.replace('.pth', f'_step{current_step}.pth')
                self.save_model(checkpoint_path)
        
        pbar.close()
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total steps: {self.total_steps}")
        print(f"Total transitions: {self.total_transitions:,}")
        print(f"Total episodes: {self.episodes_completed:,}")
        print(f"Best score: {best_score:.1f}")
        
        # Save final model
        self.save_model(save_path)
        self.plot_training_curves()
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent without exploration."""
        # Create a separate environment for evaluation
        eval_env = VectorizedGame2048(num_envs=1, seed=99)
        
        scores = []
        max_tiles = []
        
        for _ in range(num_episodes):
            state = eval_env.reset()[0:1]  # Single env
            done = False
            
            while not done:
                action = self.select_actions(state, training=False)
                next_state, _, dones, info = eval_env.step(action)
                done = dones[0]
                state = next_state
            
            scores.append(info['scores'][0])
            max_tile = 2 ** info['max_tiles'][0] if info['max_tiles'][0] > 0 else 0
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
            'training_history': self.training_history,
            'recent_scores': self.recent_scores[-1000:],  # Keep last 1000
            'recent_max_tiles': self.recent_max_tiles[-1000:],
            'hyperparameters': {
                'num_envs': self.num_envs,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'invalid_move_penalty': self.invalid_move_penalty,
                'use_double_dqn': self.use_double_dqn,
                'train_steps_per_collect': self.train_steps_per_collect
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
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_transitions = checkpoint.get('total_transitions', 0)
        self.episodes_completed = checkpoint.get('episodes_completed', 0)
        self.training_updates = checkpoint.get('training_updates', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        if 'recent_scores' in checkpoint:
            self.recent_scores = checkpoint['recent_scores']
        if 'recent_max_tiles' in checkpoint:
            self.recent_max_tiles = checkpoint['recent_max_tiles']
        
        # Validate hyperparameters
        if 'hyperparameters' in checkpoint:
            saved = checkpoint['hyperparameters']
            current = {
                'num_envs': self.num_envs,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'use_double_dqn': self.use_double_dqn
            }
            
            for key in current:
                if key in saved and saved[key] != current[key]:
                    print(f"WARNING: {key} mismatch: saved={saved[key]}, current={current[key]}")
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Steps: {self.total_steps}, Episodes: {self.episodes_completed}")
        print(f"  Epsilon: {self.epsilon:.4f}")
    
    def plot_training_curves(self, save_path: str = "plots/training_curves_vec.png") -> None:
        """Plot training curves."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        def smooth(data, window=50):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Plot scores
        if self.training_history['step_scores']:
            axes[0, 0].plot(smooth(self.training_history['step_scores']))
            axes[0, 0].set_title('Average Score')
            axes[0, 0].set_xlabel('Log Step')
            axes[0, 0].grid(True)
        
        # Plot rewards
        if self.training_history['step_rewards']:
            axes[0, 1].plot(smooth(self.training_history['step_rewards']))
            axes[0, 1].set_title('Average Reward per Transition')
            axes[0, 1].set_xlabel('Log Step')
            axes[0, 1].grid(True)
        
        # Plot max tiles
        if self.training_history['max_tiles']:
            axes[0, 2].plot(self.training_history['max_tiles'])
            axes[0, 2].set_title('Max Tile Achieved')
            axes[0, 2].set_xlabel('Log Step')
            axes[0, 2].grid(True)
        
        # Plot losses
        losses = [l for l in self.training_history['losses'] if l > 0]
        if losses:
            axes[1, 0].plot(smooth(losses))
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Log Step')
            axes[1, 0].grid(True)
        
        # Plot epsilon
        if self.training_history['epsilons']:
            axes[1, 1].plot(self.training_history['epsilons'])
            axes[1, 1].set_title('Epsilon')
            axes[1, 1].set_xlabel('Log Step')
            axes[1, 1].grid(True)
        
        # Plot throughput
        if self.training_history['transitions_per_sec']:
            axes[1, 2].plot(smooth(self.training_history['transitions_per_sec']))
            axes[1, 2].set_title('Transitions/sec')
            axes[1, 2].set_xlabel('Log Step')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training curves saved to {save_path}")


def main(
    resume_from: str = None,
    num_envs: int = 16,
    total_steps: int = 50000,
    batch_size: int = 512
):
    """
    Main training function.
    
    Args:
        resume_from: Path to checkpoint to resume from
        num_envs: Number of parallel environments
        total_steps: Total environment steps to train
        batch_size: Training batch size
    """
    # Create trainer
    # Key insight: with 16 envs and step-based epsilon decay, we need MUCH slower decay
    # Old: 0.99995 per step → epsilon=0.08 after 50k steps (too fast!)
    # New: 0.999995 per step → epsilon=0.60 after 100k steps, 0.37 after 200k steps
    trainer = VectorizedDQNTrainer(
        num_envs=num_envs,
        learning_rate=5e-5,  # Reduced from 1e-4 (less aggressive updates)
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999995,  # Much slower: ~0.37 at 200k steps, ~0.14 at 400k steps
        batch_size=batch_size,
        buffer_capacity=1000000,  # Larger buffer: 1M transitions for more diversity
        target_update_freq=2000,  # Less frequent target updates (more stability)
        invalid_move_penalty=-10.0,
        use_double_dqn=True,
        train_steps_per_collect=2,  # Reduced from 4 (less overfitting)
        seed=42
    )
    
    # Resume if provided
    if resume_from:
        print(f"\nResuming from: {resume_from}")
        trainer.load_model(resume_from)
    
    # Train
    trainer.train(
        total_steps=total_steps,
        log_freq=100,
        eval_freq=500,
        save_freq=1000,
        save_path="models/dqn_2048_vec.pth"
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation (20 episodes)")
    print("="*60)
    final_stats = trainer.evaluate(num_episodes=20)
    print(f"Average Score: {final_stats['avg_score']:.1f} ± {final_stats['std_score']:.1f}")
    print(f"Max Score: {final_stats['max_score']}")
    print(f"Median Max Tile: {final_stats['median_max_tile']}")
    print(f"Best Max Tile: {final_stats['max_tile']}")
    
    with open('models/final_stats_vec.json', 'w') as f:
        json.dump(final_stats, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vectorized 2048 DQN Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num-envs', type=int, default=16,
                        help='Number of parallel environments (default: 16)')
    parser.add_argument('--steps', type=int, default=200000,
                        help='Total environment steps (default: 200000, ~3.2M transitions)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size (default: 256)')
    
    args = parser.parse_args()
    
    main(
        resume_from=args.resume,
        num_envs=args.num_envs,
        total_steps=args.steps,
        batch_size=args.batch_size
    )
