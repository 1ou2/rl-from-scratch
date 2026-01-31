from venv import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    logger.info("GPU not available, using CPU")

OBS_SIZE = 4
ACTION_SIZE = 2
HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70.0

class SimpleNet(nn.Module):
    """A simple neural network
    - input is a tensor of observations
    - hidden layer
    - output of action size (to select the next action)"""
    def __init__(self,obs_size:int,action_size:int,hidden_size:int=HIDDEN_SIZE):
        super(SimpleNet,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        self.to(DEVICE)

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
    max_steps = 500
    step_count = 0
    while( not terminated and not truncated and step_count < max_steps): 
        logger.debug(f"{obs} {type(obs)}")
        t_obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        action_logits = net(t_obs)
        logger.debug(f"{action_logits}")
        action_prob = torch.softmax(action_logits, dim=-1)
        logger.debug(f"{action_prob}")
        # sample action
        action = torch.multinomial(action_prob, num_samples=1).item()
        logger.debug(f"sampled action {action}")
        steps.append(EpisodeStep(observation=t_obs, action=action))
        obs, reward, terminated, truncated,info = env.step(action)
        logger.debug(f"reward {reward} terminated {terminated} truncated {truncated} info {info}")
        total_reward += reward
        step_count += 1
    return Episode(reward=total_reward, steps=steps)
        
def filter_top_episodes(episodes:list[Episode], percentile:float=PERCENTILE):
    """Filters the top episodes based on reward percentile"""
    rewards = [episode.reward for episode in episodes]
    reward_bound = np.percentile(rewards,percentile)
    top_episodes = [episode for episode in episodes if episode.reward >= reward_bound]
    return top_episodes, reward_bound

def init_env():
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    act = env.action_space.sample() 
    logger.info(f"obs space {env.observation_space}")
    logger.info(f"act space {env.action_space}")
    logger.debug(f"sample action {act}") # take a random action
    logger.debug(f"sample obs {obs}")
    logger.info(f"info {info}")

    return env
    
if __name__ == "__main__":
    env = init_env()
    net = SimpleNet(OBS_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    print(f"{'-'*20} Episode Result {'-'*20}")

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    max_training_iterations = 1000
    iteration = 0

    while iteration < max_training_iterations:
        iteration += 1
        episodes = []
        for _ in range(BATCH_SIZE):
            episode = generate_episode(env, net)
            episodes.append(episode)
        top_episodes, reward_bound = filter_top_episodes(episodes, PERCENTILE)
        print(f"{iteration}  - Reward bound for top {PERCENTILE} percentile: {reward_bound}")
        if reward_bound >= 500:
            logger.info("Solved!")
            break
        # Batch process all observations and actions for efficiency
        all_observations = []
        all_actions = []
        for episode in top_episodes:
            for step in episode.steps:
                all_observations.append(step.observation)
                all_actions.append(step.action)
        
        if all_observations:  # Only process if there are steps
            # Stack tensors and move to device (already on device from generate_episode)
            obs_batch = torch.stack(all_observations)
            actions_batch = torch.tensor(all_actions, device=DEVICE, dtype=torch.long)
            
            # Forward pass
            logits = net(obs_batch)
            
            # Compute loss in batch
            loss = F.cross_entropy(logits, actions_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    env.close()
    logger.info("Training complete!")