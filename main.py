import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os
from gym.envs.registration import register


register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

env_id = 'FrozenLakeNotSlippery-v0'
env = gym.make(env_id)

"""
use gym (openAI)
https://blog.techbridge.cc/2017/11/04/openai-gym-intro-and-q-learning/

"""

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

print(env.observation_space.n)
print(env.action_space.n)

epsilon_start = 1.
epsilon_final = 0.01
epsilon_decay = 3000.

def epsilon_by_frame(frame_idx):
    epsilon = 1 / frame_idx;
    return epsilon




plt.plot([epsilon_by_frame(i) for i in range(100000)])

def act(state, epsilon):
    action = 0
    max = Q[state][0];
    if random.random() > epsilon:
        for i in range (1, 4):
            if Q[state][i] > max:
                action = i;
    else:
        action = random.randint(0, 3)

    return action


# # Start learning

# In[1]:


Q = np.zeros((16, 4))
losses         = []
all_rewards    = []
frames = []
episode_reward = 0
num_frames = 10000
gamma = 0.8
rate = 0.9
count = 0
state = env.reset()
for frame_idx in range(1, num_frames + 1):
    # get epsilon
    epsilon = epsilon_by_frame(frame_idx)

    # forward
    action  = act(state, epsilon)

    # interact with environment
    env.render()
    next_state, reward, done, info = env.step(action)
    next_action = act(next_state, epsilon)

    # update Q table
    Q[state][action] = Q[state][action] + rate * (reward + gamma * Q[next_state][next_action] + Q[state][action])

    # go to next state
    state = next_state
    episode_reward += reward


    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        print('-----------done')
        if (reward == 1): count += 1

env.close()
print (count)
print(Q)
