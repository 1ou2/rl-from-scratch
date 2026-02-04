"""
Cross-Entropy Method (CEM) for 2048 - Alternative Training Strategy
This is simpler but less efficient than DQN. Provided for comparison.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from tqdm import tqdm

from game_2048 import Game2048
from neural_network import DQN2048


class CEMTrainer:
    """
    Cross-Entropy Method trainer for 2048.
    
    Simpler than DQN but less sample-efficient.
    Good for quick prototyping or baseline comparison.
    """
    
    def __init__(
        self,
        env: Game2048,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        elite_frac: float = 0.2,
        device: str = None
    ):
        """
        Initialize CEM trainer.
        
        Args:
            env: Game2048 environment
            learning_rate: Learning rate
            batch_size: Number of episodes per iteration
            elite_frac: Fraction of top episodes to use for training
            device: Device to use
        """
        self.env = env
        self.batch_size = batch_size
        self.elite_frac = elite_frac
        self.n_elite = max(1, int(batch_size * elite_frac))
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.policy_net = DQN2048().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def generate_episode(self, exploration_noise: float = 0.1) -> Tuple[List, float, int]:
        """
        Generate one episode using current policy.
        
        Args:
            exploration_noise: Probability of random action
            
        Returns:
            (trajectory, total_reward, max_tile)
            trajectory is list of (state, action) pairs
        """
        state, _ = self.env.reset()
        trajectory = []
        total_reward = 0
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            if np.random.random() < exploration_noise:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()
            
            # Store state-action pair
            trajectory.append((state.copy(), action))
            
            # Execute action
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Only count reward if move was valid
            if info['grid_changed']:
                total_reward += reward
            
            state = next_state
        
        max_tile = 2 ** state.max() if state.max() > 0 else 0
        
        return trajectory, total_reward, max_tile
    
    def train_iteration(self, exploration_noise: float = 0.1) -> dict:
        """
        Perform one CEM training iteration.
        
        1. Generate batch of episodes
        2. Select elite episodes
        3. Train on elite trajectories
        
        Returns:
            Statistics dictionary
        """
        # Generate batch of episodes
        episodes = []
        for _ in range(self.batch_size):
            trajectory, reward, max_tile = self.generate_episode(exploration_noise)
            episodes.append({
                'trajectory': trajectory,
                'reward': reward,
                'max_tile': max_tile
            })
        
        # Sort by reward and select elite episodes
        episodes.sort(key=lambda x: x['reward'], reverse=True)
        elite_episodes = episodes[:self.n_elite]
        
        # Extract states and actions from elite episodes
        elite_states = []
        elite_actions = []
        
        for episode in elite_episodes:
            for state, action in episode['trajectory']:
                elite_states.append(state)
                elite_actions.append(action)
        
        # Train on elite trajectories
        if len(elite_states) > 0:
            states_tensor = torch.FloatTensor(np.array(elite_states)).to(self.device)
            actions_tensor = torch.LongTensor(elite_actions).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            q_values = self.policy_net(states_tensor)
            
            # Cross-entropy loss (train to predict elite actions)
            loss = self.criterion(q_values, actions_tensor)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            loss_value = loss.item()
        else:
            loss_value = 0.0
        
        # Compute statistics
        rewards = [ep['reward'] for ep in episodes]
        max_tiles = [ep['max_tile'] for ep in episodes]
        elite_rewards = [ep['reward'] for ep in elite_episodes]
        
        stats = {
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'elite_mean_reward': np.mean(elite_rewards),
            'mean_max_tile': np.mean(max_tiles),
            'max_tile': np.max(max_tiles),
            'loss': loss_value
        }
        
        return stats
    
    def train(self, num_iterations: int = 500, exploration_noise: float = 0.1):
        """
        Train using CEM for multiple iterations.
        
        Args:
            num_iterations: Number of training iterations
            exploration_noise: Initial exploration noise
        """
        print(f"\nTraining with Cross-Entropy Method")
        print(f"Batch size: {self.batch_size}")
        print(f"Elite fraction: {self.elite_frac} ({self.n_elite} episodes)")
        print(f"Exploration noise: {exploration_noise}")
        print()
        
        best_reward = 0
        
        for iteration in tqdm(range(num_iterations), desc="CEM Training"):
            # Decay exploration
            noise = exploration_noise * (0.995 ** iteration)
            
            # Train iteration
            stats = self.train_iteration(noise)
            
            if stats['max_reward'] > best_reward:
                best_reward = stats['max_reward']
            
            # Print progress
            if (iteration + 1) % 50 == 0:
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"  Mean Reward: {stats['mean_reward']:.2f}")
                print(f"  Elite Reward: {stats['elite_mean_reward']:.2f}")
                print(f"  Max Reward: {stats['max_reward']:.2f}")
                print(f"  Mean Max Tile: {stats['mean_max_tile']:.0f}")
                print(f"  Best Reward: {best_reward:.2f}")
                print(f"  Exploration Noise: {noise:.4f}")
        
        print(f"\nTraining complete! Best reward: {best_reward:.2f}")


def compare_strategies():
    """
    Quick comparison of CEM vs DQN approaches.
    """
    print("="*60)
    print("COMPARISON: Cross-Entropy Method vs Deep Q-Learning")
    print("="*60)
    
    print("\n1. CROSS-ENTROPY METHOD (CEM)")
    print("-" * 60)
    print("✓ Simpler to implement")
    print("✓ No need for replay buffer or target network")
    print("✓ Good for quick prototyping")
    print("✗ Needs full episodes (slower)")
    print("✗ Wastes data from non-elite episodes")
    print("✗ Less sample efficient")
    print(f"✗ Typical: {16 * 500} = 8,000 episodes for training")
    
    print("\n2. DEEP Q-LEARNING (DQN)")
    print("-" * 60)
    print("✓ Learns from every transition")
    print("✓ Experience replay improves sample efficiency")
    print("✓ Can penalize invalid moves immediately")
    print("✓ Target network stabilizes learning")
    print("≈ More complex (but train_2048.py implements it)")
    print(f"✓ Typical: 5,000 episodes, but learns from ~500k transitions")
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Use DQN (train_2048.py)")
    print("="*60)
    print("\nCEM is provided here for:")
    print("  1. Educational comparison")
    print("  2. Quick baseline testing")
    print("  3. Understanding different RL paradigms")
    print("\nBut DQN will give you better results for 2048!")
    print("="*60)


if __name__ == "__main__":
    compare_strategies()
    
    # Quick CEM demo (will not perform as well as DQN)
    print("\n\nRunning quick CEM demo (100 iterations)...")
    print("Note: This is just a demonstration. Use DQN for actual training.")
    
    env = Game2048()
    trainer = CEMTrainer(env, batch_size=8, elite_frac=0.25)
    trainer.train(num_iterations=100, exploration_noise=0.2)
