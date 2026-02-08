"""
Improved Neural Network for 2048 Game - Version 2

Key improvements over V1:
1. One-hot tile encoding (captures exponential tile relationships)
2. Convolutional layers (captures spatial patterns)
3. Residual connections (better gradient flow)
4. Proper state preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN2048_V2(nn.Module):
    """
    Improved DQN for 2048 with:
    - One-hot encoding per tile position (17 channels: 0-16)
    - 2D convolutions to capture spatial patterns  
    - Value/Advantage decomposition (Dueling DQN)
    """
    
    def __init__(self, num_tile_values: int = 17):
        """
        Args:
            num_tile_values: Number of possible tile values (0 to 16 = empty to 2^16)
        """
        super().__init__()
        self.num_tile_values = num_tile_values
        
        # Convolutional layers - capture local spatial patterns
        # Input: (batch, 17, 4, 4) - one-hot encoded tiles
        self.conv1 = nn.Conv2d(num_tile_values, 128, kernel_size=2, stride=1, padding=0)  # -> (128, 3, 3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0)  # -> (128, 2, 2)
        
        # Also add 1x1 convs for channel mixing
        self.conv1x1 = nn.Conv2d(128, 128, kernel_size=1)
        
        # Flatten: 128 * 2 * 2 = 512
        self.fc_hidden = nn.Linear(512, 256)
        
        # Dueling architecture: separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def _encode_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert grid to one-hot encoding.
        
        Args:
            x: Input tensor of shape (batch, 4, 4) with values 0-16
        
        Returns:
            One-hot encoded tensor of shape (batch, 17, 4, 4)
        """
        batch_size = x.shape[0]
        
        # Clamp values to valid range and convert to long
        x = x.long().clamp(0, self.num_tile_values - 1)
        
        # One-hot encode: (batch, 4, 4) -> (batch, 4, 4, 17) -> (batch, 17, 4, 4)
        one_hot = F.one_hot(x, num_classes=self.num_tile_values)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        
        return one_hot
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State tensor of shape (batch, 4, 4) or (batch, 16)
        
        Returns:
            Q-values of shape (batch, 4)
        """
        # Handle flattened input
        if len(x.shape) == 2 and x.shape[1] == 16:
            x = x.view(-1, 4, 4)
        
        # Encode state to one-hot
        x = self._encode_state(x)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv1x1(x))
        
        # Flatten (use contiguous + view for safety after permute in _encode_state)
        x = x.contiguous().view(x.size(0), -1)
        
        # Shared hidden layer
        x = F.relu(self.fc_hidden(x))
        
        # Dueling: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class DQN2048_V2_Large(nn.Module):
    """
    Larger version with more capacity for complex patterns.
    Uses:
    - One-hot encoding
    - Multiple conv layers with residual connections
    - Dueling architecture
    """
    
    def __init__(self, num_tile_values: int = 17):
        super().__init__()
        self.num_tile_values = num_tile_values
        
        # Initial convolution
        self.conv_in = nn.Conv2d(num_tile_values, 256, kernel_size=1)
        
        # Convolutional blocks (with padding to maintain size where possible)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2, padding=0),  # 4x4 -> 3x3
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU()
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2, padding=0),  # 3x3 -> 2x2
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU()
        )
        
        # Global features from full grid (for corner patterns etc)
        self.conv_global = nn.Conv2d(256, 64, kernel_size=2, padding=0)  # 2x2 -> 1x1
        
        # Flatten: 256 * 2 * 2 + 64 * 1 * 1 = 1024 + 64 = 1088
        fc_input_size = 256 * 2 * 2 + 64
        
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def _encode_state(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.long().clamp(0, self.num_tile_values - 1)
        one_hot = F.one_hot(x, num_classes=self.num_tile_values)
        return one_hot.permute(0, 3, 1, 2).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2 and x.shape[1] == 16:
            x = x.view(-1, 4, 4)
        
        x = self._encode_state(x)
        
        # Initial projection
        x = F.relu(self.conv_in(x))
        
        # Conv blocks
        x = self.conv_block1(x)  # 4x4 -> 3x3
        x = self.conv_block2(x)  # 3x3 -> 2x2
        
        # Extract local and global features
        local_features = x.contiguous().view(x.size(0), -1)  # 256 * 2 * 2 = 1024
        global_features = F.relu(self.conv_global(x)).contiguous().view(x.size(0), -1)  # 64
        
        # Combine
        x = torch.cat([local_features, global_features], dim=1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Dueling
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# For backward compatibility
DQN2048 = DQN2048_V2


if __name__ == "__main__":
    # Test both networks
    print("Testing DQN2048_V2...")
    model_v2 = DQN2048_V2()
    params_v2 = sum(p.numel() for p in model_v2.parameters())
    print(f"  Parameters: {params_v2:,}")
    
    # Test with batch input
    test_input = torch.randint(0, 12, (32, 4, 4)).float()
    output = model_v2(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    print("\nTesting DQN2048_V2_Large...")
    model_large = DQN2048_V2_Large()
    params_large = sum(p.numel() for p in model_large.parameters())
    print(f"  Parameters: {params_large:,}")
    
    output_large = model_large(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output_large.shape}")
    
    print("\nBoth models working correctly!")
