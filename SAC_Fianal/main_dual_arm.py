
#  双机械臂训练脚本

import argparse
import datetime
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
import matplotlib.pyplot as plt
from dual_arm import Arm
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="dual_arm",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=20000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=120*500, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=120*1000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = Arm()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
episode_num = 2000
load = 1
#TesnorboardX
writer = SummaryWriter(logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))
# Memory
memory = ReplayMemory(args.replay_size)
ounoise = OUNoise(env.action_space.shape[0])
# Training Loop
total_numsteps = 0
updates = 0
noise_scale = 2.0
final_noise_scale=0.3
rewards = []
if not load:
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        ounoise.scale = (noise_scale - final_noise_scale) * max(0, 500- i_episode) / 500 + final_noise_scale
        ounoise.reset()
        while True:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()   # Sample random action
                action += ounoise.noise()
            else:
                action = agent.select_action(state)  # Sample action from policy
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
            next_state, reward, _, done= env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            mask = 1 if episode_steps == 120 else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory
            state = next_state
            if done or episode_steps==120:
                break
        rewards.append(episode_reward)
        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
        if i_episode == episode_num:
            break
    ep_r_hist = np.reshape(rewards, [-1, 1])
    plt.figure(1)
    plt.plot(ep_r_hist)
    plt.savefig('rewards_dual_arm_Gaussian .png')
    agent.save_model(args.env_name, "Gaussian")
else:
    agent.load_model("./models/sac_actor_dual_arm_Gaussian",
                     "./models/sac_critic_dual_arm_Gaussian")
    for i in range(6):
        state = env.reset()
        t = 0
        ep_reward = 0
        while True:
            action = agent.select_action(state,evaluate=True)
            next_state, reward, _, done = env.step(action)
            t += 1
            ep_reward += reward
            state = next_state
            if done or t % 120== 0:
                print(ep_reward)
                break
