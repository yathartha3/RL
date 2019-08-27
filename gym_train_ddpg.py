# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ddpg import DDPG
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition

import torch
import pickle
import numpy as np

import random
import copy
import rclpy
from tensorboardX import SummaryWriter
import gym
import mod_utils as utils


class Parameters:
    def __init__(self):
        #DDPG params
        self.use_ln = True
        self.gamma = 0.99
        self.tau = 0.001
        self.seed = 7
        self.batch_size = 128
        self.buffer_size = 1000000
        self.frac_frames_train = 1
        self.use_done_mask = True
        self.num_evals = 1
        self.num_episodes = 1000
        self.time_steps = 100

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = 'RL/'


def trainModel(parameters, env):
    logger = SummaryWriter()
    state = env.reset()     # Required to populate observation space. # TODO: fix this
    action_size = env.action_space.shape[0]

    # define the reinforcement learning agent
    agent = DDPG(parameters)

    # Define replay buffer and noise
    replay_buffer = ReplayMemory(parameters.buffer_size)
    ounoise = OUNoise(parameters.action_dim)

    episode_rewards_list = []
    validation_mean_rewards_list = []

    # Start of episodes
    for episode in range(parameters.num_episodes):
        print("Episode = " + str(episode))
        state = torch.Tensor([env.reset()])

        episode_reward = 0.0
        done = False
        # Start of time-steps
        while not done:
            action = agent.actor.forward(state)
            # Clamp the actions to that of what the env can use
            action.clamp(env.action_space.low[0], env.action_space.high[0])
            action = utils.to_numpy(action)

            is_action_noise = True
            if is_action_noise: action += ounoise.noise()

            next_state, reward, done, info = env.step(action.flatten())
            if render:
                env.render()
            #print("Action = " + str(action))
            #print("reward = " + str(reward))

            episode_reward += reward

            action = torch.Tensor(action)
            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            replay_buffer.push(state, action, next_state, reward, mask)
            state = copy.deepcopy(next_state)

        logger.add_scalar('Pendulum/episode_rewards', episode_reward, episode)
        print("Episode ended.")

        if len(replay_buffer) > parameters.batch_size * 5:
            for _ in range(parameters.frac_frames_train):
                transitions = replay_buffer.sample(parameters.batch_size)
                batch = Transition(*zip(*transitions))
                agent.update_parameters(batch)

        # end of this episode
        episode_rewards_list.append(episode_reward)

        # Run validation on the current policy
        for n in range(3):      # TODO: make number of validation episodes param
            state = torch.Tensor([env.reset()])
            test_episode_ddpg_reward = 0.0
            done = False
            while not done:
                action = agent.actor.forward(state)
                action = utils.to_numpy(action)
                next_state, reward, done, info = env.step(action.flatten())#env.step(action.numpy()[0])
                test_episode_ddpg_reward += reward
                next_state = torch.Tensor([next_state])
                state = copy.deepcopy(next_state)
                if done:
                    break

        test_episode_ddpg_reward = np.mean(test_episode_ddpg_reward)
        validation_mean_rewards_list.append(test_episode_ddpg_reward)
        print("\nValidation Reward = " + str(test_episode_ddpg_reward))

        print("Training: Episode: {}, noise: {}, reward: {}, average reward: {}".format(episode, ounoise.scale,
                                                                                        episode_rewards_list[-1],
                                                                                        np.mean(episode_rewards_list[-10:])))
        print("\n")
        torch.save(agent.actor.state_dict(), 'pendulum_model.pt')

    pickling_on = open("pendulum_rewards.p", "wb")
    pickle.dump(validation_mean_rewards_list, pickling_on)
    pickling_on.close()
    logger.close()


def main(args=None):
    parameters = Parameters()
    # Initialize the environment
    env_tag = 'Pendulum-v0'

    #Create Env
    env = gym.make(env_tag)
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Use manual seed
    #Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)

    trainModel(parameters, env)


if __name__ == "__main__":
    render = False
    main()
