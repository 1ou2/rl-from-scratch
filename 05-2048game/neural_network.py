"""
Neural Network Implementation for 2048 Game using PyTorch
This module provides a fully connected neural network for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN2048(nn.Module):
    """
    Deep Q-Network for 2048 Game using fully connected layers.
    
    Architecture:
    - Input: 16 values (flattened 4x4 grid)
    - Hidden layers: 512 -> 256 -> 128 neurons (increased capacity)
    - Output: 4 Q-values (one for each action: up, down, left, right)
    - Activation: ReLU
    - Includes batch normalization and dropout for better training stability
    """
    
    def __init__(self, input_size: int = 16, output_size: int = 4):
        """
        Initialize the neural network.
        
        Args:
            input_size: Size of input (default 16 for 4x4 grid)
            output_size: Number of actions (default 4 for up/down/left/right)
        """
        super(DQN2048, self).__init__()
        
        # Define the fully connected layers with increased capacity
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.05)
        
        self.fc4 = nn.Linear(128, output_size)
        
        # Initialize weights using Xavier uniform initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
        # Initialize batch norm layers
        for bn in [self.bn1, self.bn2, self.bn3]:
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4, 4) or (batch_size, 16)
            
        Returns:
            Q-values tensor of shape (batch_size, 4)
        """
        # Flatten input if it's in 4x4 format
        if len(x.shape) == 3:  # (batch_size, 4, 4)
            x = x.view(x.size(0), -1)
        elif len(x.shape) == 2 and x.shape[1] == 16:  # Already flattened
            pass
        else:
            raise ValueError(f"Invalid input shape: {x.shape}. Expected (batch_size, 4, 4) or (batch_size, 16)")
        
        # Normalize input: scale from 0-16 range to 0-1
        x = x / 16.0
        
        # Forward pass through layers with ReLU activation and batch norm
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)  # No activation on output layer for Q-values
        
        return x
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Game state tensor
            epsilon: Exploration probability (0.0 = greedy, 1.0 = random)
            
        Returns:
            Action index (0=up, 1=down, 2=left, 3=right)
        """
        if torch.rand(1).item() < epsilon:
            # Random action
            return torch.randint(0, 4, (1,)).item()
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0) if len(state.shape) == 2 else state)
                return q_values.argmax(dim=1).item()


# Example usage and testing
if __name__ == "__main__":
    # Create network
    model = DQN2048()
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test with different input formats
    print("\nTesting network...")
    
    # Test with 4x4 input
    batch_size = 32
    test_input_4x4 = torch.randn(batch_size, 4, 4)
    output = model(test_input_4x4)
    print(f"Input shape (4x4): {test_input_4x4.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with flattened input
    test_input_flat = torch.randn(batch_size, 16)
    output_flat = model(test_input_flat)
    print(f"Input shape (flat): {test_input_flat.shape}")
    print(f"Output shape: {output_flat.shape}")
    
    # Test single state action selection
    single_state = torch.randn(4, 4)
    action_greedy = model.get_action(single_state, epsilon=0.0)
    action_random = model.get_action(single_state, epsilon=1.0)
    print(f"\nGreedy action: {action_greedy}")
    print(f"Random action: {action_random}")
    
    # Print model architecture
    print(f"\nModel architecture:")
    print(f"Input -> FC(16, 256) -> ReLU -> FC(256, 128) -> ReLU -> FC(128, 64) -> ReLU -> FC(64, 4) -> Output")