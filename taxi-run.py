#import libraries
import numpy as np
import gym
import random
import pickle

from tqdm import *

with open('./taxi.pickle', 'rb') as f:
   q_table = pickle.load(f)

env = gym.make('Taxi-v3')

stop = False
state = env.reset()

rewards = []

while not stop:
   env.render()
   action = np.argmax(q_table[state])
   state, reward, stop, _ = env.step(action)
   rewards.append(reward)

print('last reward: ', rewards[-1])

env.close()
