import warnings

import __init__

import gym
import gym_2048  # there may be problems during import. check __init__.py in this directory for more.
import pygame
from cmdargs import args
from output import Output

import time
from collections import Counter

import gym
import pygame
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import random
from cmdargs import args

import tensorflow as tf

"""
do the following to load a model from '.....h5' and save the evaluated results to '....png':
     python snowfight-starter/snowfight_qtest.py -f 'qtable_202212190424/model_epoch9999.h5' 
                                                 -o 'evaluation_202212190424e9999.png' -e 20

"""

warnings.filterwarnings("ignore")

size = args.size
mode = args.mode
ws   = args.window_size

# for random agent, use play v1 script with -m human_rand
assert mode != "human_rand"

window_size = args.window_size
max_steps = args.max_steps

# DQN model
model = tf.keras.models.load_model(args.file, compile=False)
print(model.summary())

env = gym.make('gym_2048/Typical2048', render_mode=mode, size=size, window_size=window_size, max_enemies=args.enemies)
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

if args.fps is not None:
    env.metadata["render_fps"] = args.fps
else:
    env.metadata["render_fps"] = 1000000000

env.action_space.seed(args.seed)

output = Output(args.output_file, 'model', output_every_n=args.episodes, random_overlay=True)  # does any output job.
folder = output.dir if args.output_file is not None else None

# Test the agent
test_episodes = args.episodes
max_steps = args.max_steps

episode = 0
total_score = 0
max_score = 0
total_steps = 0
min_steps_achieved = (2 << 15)
max_steps_achieved = 0

running = True
step = 0


def end():
    if episode > 0:
        print(f"Average score: {total_score / episode:.2f}\n" +
              f"Maximum score: {max_score:.2f}\n" +
              f"Average steps: {total_steps / episode:.2f} ([{min_steps_achieved} to {max_steps_achieved}])")

    env.close()


def vectorize(inp: np.ndarray):
    return np.expand_dims(inp, axis=0)


print("Testing started ...")
for episode in range(test_episodes):
    for ii in range(2 if args.mode != 'human' else 1):
        state = env.reset(seed=args.seed)[0]  # [0] for observation only
        total_testing_rewards = 0

        for step in range(max_steps):
            if args.mode != 'rgb_array':
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        end()
                        exit()

            if ii == 0:
                "Obtain Q-values from network."
                q_values = model(vectorize(state))

                "Select action based on q-value."
                action = np.argmax(q_values[0])
            else:
                action = env.action_space.sample()

            "Deterimine next state."
            new_state, reward, done, truncated, info = env.step(action)  # take action and get reward
            state = new_state

            # print(state, action)
            if done or truncated:
                if done:
                    if ii == 0:
                        total_score += info['score']
                        max_score = max(max_score, info['score'])

                        output.log(done, episode, step, info, model)

                        total_steps += step
                        max_steps_achieved = max(max_steps_achieved, step)
                        min_steps_achieved = min(min_steps_achieved, step)
                    else:
                        output.concat({'episode': [episode], 'best': [info['score']], 'good': [step]})
                    break

end()
