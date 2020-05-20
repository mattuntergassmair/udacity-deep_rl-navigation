#!/usr/bin/env python

# Project 1: Navigation
# Udacity Nanodegree: Deep Reinforcement Learning
# https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893

import numpy as np
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from qlearning.random_agent import RandomAgent
from qlearning.dqn_agent import DQNAgent
from qlearning.simulation import simulate, moving_average  # , train


def main():
    """
    Collect yellow bananas, avoid blue ones
    """
    # file_name pointing to Unity environment
    rnd_seed = 42
    np.random.seed(rnd_seed)
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", seed=rnd_seed)  # with visualization
    # env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64", seed=rnd_seed)  # no visualization (about 5% faster)

    # Environments contain brains which are responsible
    # for deciding the actions of their associated agents.
    brain_name = env.brain_names[0]  # get the default brain name
    env_info = env.reset(train_mode=True)[brain_name]

    # 2. Examine the State and Action Spaces

    # The simulation contains a single agent that navigates a large environment.
    # At each time step, it has four actions at its disposal:
    # - `0` - walk forward
    # - `1` - walk backward
    # - `2` - turn left
    # - `3` - turn right
    # The state space has `37` dimensions and contains the agent's velocity,
    # along with ray-based perception of objects around agent's forward direction.
    # A reward of `+1` is provided for collecting a yellow banana,
    # and a reward of `-1` is provided for collecting a blue banana.

    # Print basic environment info
    action_size = env.brains[brain_name].vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    print('Number of agents: ', len(env_info.agents))
    print('Number of actions:', action_size)
    print('States have length:', state_size)

    # define agents
    # layer_sizes = [state_size, state_size, 18, 8, action_size]
    # layer_sizes = [4*state_size, 2*state_size, state_size, state_size, 16, 8]
    # layer_sizes = [1024, 1024]
    layer_sizes = [2*state_size, state_size, 16, 8]
    agents = {
        'random': RandomAgent(action_size),
        'dqn':    DQNAgent(state_size, action_size, layer_sizes)
    }

    # TODO: using multiple brain names
    # TODO: save model to disk, load from disk
    # TODO: reward boxplot
    # TODO: make sure all random generators are seeded
    #
    # Improvements:
    #  - TODO: Dropout Layers? batch normalization?
    #  - TODO: adaptive learning rates (alpha, tau, epsilon)
    #  - TODO: implement improved Q-learning (rainbow)

    # TODO: n_episodes vs n_epochs
    scores = simulate(env, agents['dqn'], brain_name, learn=True, n_episodes=800) # 1400)

    window_size = 32
    plt.plot(
        range(window_size, len(scores)+1),
        moving_average(scores, window_size)
    )
    plt.title("Reward per episode (running mean)")
    plt.xlabel("number of episodes")
    plt.ylabel("total reward")
    plt.draw()

    for key in agents:
        simulate(env, agents[key], brain_name, learn=False, n_episodes=5)

    env.close()

    plt.show()


if __name__ == "__main__":
    main()
