import torch
import torch.nn as nn
import torch.functional as F
from torch import tensor
import gymnasium as gym
from dataclasses import dataclass

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

@dataclass
class EpisodeStep:
    observation: tensor
    action: int

@dataclass
class Episode:
    reward: float
    steps: list[EpisodeStep]


def generate_episode(env:gym.Env, net:SimpleNet):
    terminated,truncated = False,False
    total_reward = 0.0
    obs, info = env.reset()
    steps = []
    while( not terminated and not truncated): 
        print(f"{obs} {type(obs)}")
        t_obs = torch.tensor(obs,dtype=torch.float32)
        action_logits = net(t_obs)
        print(f"{action_logits}")
        action_prob = torch.softmax(action_logits,dim=-1)
        print(f"{action_prob}")
        # sample action
        action = torch.multinomial(action_prob,num_samples=1).item()
        print(f"sampled action {action}")
        steps.append(EpisodeStep(observation=t_obs, action=action))
        obs, reward, terminated, truncated,info = env.step(action)
        print(f"reward {reward} terminated {terminated} truncated {truncated} info {info}")
        total_reward += reward

    return Episode(reward=total_reward, steps=steps)
        

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
    print(f"{'-'*20} Episode Result {'-'*20}")
    for _ in range(10):
        episode = generate_episode(env,net)

        print(f"Episode reward: {episode.reward}")

