import torch
import torch.nn as nn
import torch.functional as F
import gymnasium as gym

class SimpleNet(nn.Module):
    """A simple neural network
    - input is a tensor of observations
    - hidden layer
    - output of action size (to select the next action)"""
    def __init__(self,obs_size:int,action_size:int,hidden_size:int=128):
        super(SimpleNet,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self,x:torch.Tensor):
        return self.net(x)
    
def init_env():
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    print(f"obs {obs}")
    print(f"info {info}")
    
if __name__ == "__main__":
    init_env()