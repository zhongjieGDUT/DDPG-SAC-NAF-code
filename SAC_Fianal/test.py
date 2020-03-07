from torch.distributions import Normal
import torch
import gym
import numpy as np
from torch.autograd import Variable
import os
env = gym.make('MountainCarContinuous-v0')

device = torch.device("cuda:0")
print(device)
