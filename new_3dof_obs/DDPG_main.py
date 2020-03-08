import argparse
# import math
# from collections import namedtuple
# from itertools import count
# from tqdm import tqdm
import numpy as np
import pandas as pd
from dual_arm import Arm
import matplotlib.pyplot as plt
import torch
from ddpg import DDPG
from naf import NAF
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
load=1
#  永远不用NAF了，太菜了
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--algo', default='DDPG',
                    help='algorithm to use: DDPG | NAF')
parser.add_argument('--env-name', default="dual_arms",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--noise_scale', type=float, default=2, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.1, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=2000, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=120, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=5000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of hidden_size(default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
args = parser.parse_args()

# env = NormalizedActions(gym.make(args.env_name))
env = Arm()
# env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.algo == "NAF":
    agent = NAF(args.gamma, args.tau, args.hidden_size,
                      env.observation_space.shape[0], env.action_space.shape[0])
else:
    agent = DDPG(args.gamma, args.tau, args.hidden_size,
                      env.observation_space.shape[0],env.action_space)

memory = ReplayMemory(args.replay_size)
ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,
    desired_action_stddev=args.final_noise_scale, adaptation_coefficient=1.05) if args.param_noise else None
if load:
    state = torch.Tensor([env.reset()])
    agent.load_model("./models/ddpg_actor_dual_arms_3dof_inverse","./models/ddpg_critic_dual_arms_3dof_inverse")
    state = torch.Tensor([env.reset()])

    for i in range(6):
        state = torch.Tensor([env.reset()])
        t = 0
        ep_reward = 0
        while True:
            action = agent.select_action(state, ounoise, param_noise)
            a = action.numpy()[0]
            next_state, reward, _, done = env.step(a)
            t+=1
            ep_reward+=reward
            state = torch.Tensor([next_state])
            if done or t==args.num_steps:
                print(ep_reward)
                break
    state = torch.Tensor([env.reset()])
else:
    # var = 1.5
    # VAR_MIN = 0.1
    rewards = []
    total_numsteps = 0
    updates = 0
    state = torch.Tensor([env.reset()])
    action = agent.select_action(state, ounoise, param_noise)
    a = action.numpy()[0]
    env.step(a)
    for i_episode in range(args.num_episodes):
        state = torch.Tensor([env.reset()])
        plt.ion()
        if args.ou_noise:
            ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          i_episode) / args.exploration_end + args.final_noise_scale
            ounoise.reset()
        episode_reward = 0
        t=0
        while True:
            action = agent.select_action(state, ounoise, param_noise)
            a = action.numpy()[0]
            # a = np.random.normal(a, var)
            next_state, reward, _, done= env.step(a)
            total_numsteps += 1
            t+=1
            episode_reward += reward
            action = torch.Tensor(action)
            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            memory.push(state, action, mask, next_state, reward)
            state = next_state
            if len(memory) > args.batch_size:
                for _ in range(args.updates_per_step):
                    transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = agent.update_parameters(batch)
                    updates += 1
            if done or t==args.num_steps:
                break
        rewards.append(episode_reward)
        plt.cla()
        plt.plot(rewards)
        plt.pause(0.0001)
        result = '| done' if done else '| ----'
        print('Ep:', i_episode,
              result,
              '| R: %.2f' % episode_reward,
              '| Explore: %.2f' % ounoise.scale,
              '| episode_steps: %i' % t,
              '| Total_steps: %i' % total_numsteps,
              )

        # print(
        #     "Episode: {}, total numsteps: {}, reward: {}, end: {}".format(i_episode, total_numsteps, episode_reward,"done" if done else "--"))

        # Update param_noise based on distance metric
        # if args.param_noise:
        #     episode_transitions = memory.memory[memory.position-1:memory.position]
        #     states = torch.cat([transition[0] for transition in episode_transitions], 0)
        #     unperturbed_actions = agent.select_action(states, None, None)
        #     perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)
        #
        #     ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
        #     param_noise.adapt(ddpg_dist)
        # if i_episode % 10 == 0:
        #     state = torch.Tensor([env.reset()])
        #     episode_reward = 0
        #     t=0
        #     while True:
        #         action = agent.select_action(state)
        #
        #         next_state, reward, done, _ = env.step(action.numpy()[0])
        #         episode_reward += reward
        #
        #         next_state = torch.Tensor([next_state])
        #         t+=1
        #         state = next_state
        #         if done or t==args.num_steps:
        #             break
        #
        #
        #
        #     rewards.append(episode_reward)
        #     print("Test: Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])))

    #把数据转化为exel中
    reward = np.array(rewards)
    data = pd.DataFrame(reward)
    writer = pd.ExcelWriter('NAF_dual_rewards.xlsx')# 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')
    # ‘page_1’是写入excel的sheet名
    ep_r_hist = np.reshape(reward, [-1, 1])
    # plot rewards
    plt.figure(1)
    plt.plot(ep_r_hist)
    plt.savefig('NAF_dual_obs.png')
    writer.save()
    writer.close()
    agent.save_model(args.env_name,"6DOF")
