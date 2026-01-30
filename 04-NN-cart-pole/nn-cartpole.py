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
    
def generate_episodes(env, net):
    obs, info = env.reset()
    print(f"{obs} {type(obs)}")
    t_obs = torch.tensor(obs,dtype=torch.float32)
    action_logits = net(t_obs)
    print(f"{action_logits}")


def init_env():
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    act = env.action_space.sample() 
    print(f"obs space {env.observation_space}")
    print(f"act space {env.action_space}")
    print(f"sample action {act}") # take a random action
    print(f"sample obs {obs}")
    print(f"info {info}")

    return env
    
if __name__ == "__main__":
    env = init_env()
    net = SimpleNet(4,2,128)
    generate_episodes(env,net)
