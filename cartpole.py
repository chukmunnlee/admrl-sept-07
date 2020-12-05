# import the libraries
import numpy as np
import pandas as pd
import gym
import random

from matplotlib import pyplot as plt
from tqdm import *

ALPHA=0.03
EPSILON=1
DECAY=0.99
GAMMA=1

# hyperparameters
#ALPHA = .001
#GAMMA = 1
#EPSILON = 1
#DECAY = .99

EPISODES = 500

# create the weights
weights = np.ones((2, 4))

def policy(st, w, eps):
   # shape = (2) = (2, 4) \cdot (4, 1)
   dot = np.dot(w, st)
   if random.random() > eps:
      return np.argmax(dot)
   return np.random.choice([0, 1])

def q_value(st, act, w):
   dot = np.dot(w, st)
   if act is None:
      return dot
   return dot[act]

env = gym.make('CartPole-v1')

reward_per_ep = []

#replay_buffer = []

for i in tqdm(range(EPISODES)):

   EPSILON *= DECAY
   stop = False
   state = env.reset()
   action = policy(state, weights, EPSILON)
   steps = 0

   while not stop:
      steps += 1
      #env.render()
      new_state, reward, stop, _ = env.step(action)

      # q-learning - td target, td error
      if stop:
         td_target = reward
         reward_per_ep.append(steps)
      else:
         new_action = policy(new_state, weights, EPSILON)
         """
         td_target = reward + (GAMMA * q_value(new_state, new_action, weights))
         """

         max_action = np.argmax(q_value(new_state, None, weights))
         td_target = reward + (GAMMA * q_value(new_state, max_action, weights))
         
      # shape = (4)
      td_error = (td_target - q_value(state, action, weights)) * state

      # shape (1, 4) = 1 * (4)
      weights[action] += ALPHA * td_error

      state = new_state
      action = new_action

env.close()

plt.plot(reward_per_ep)

# running average
df = pd.DataFrame(reward_per_ep)
rolling = df.rolling(5).mean().fillna(0)

plt.plot(rolling, c='r')

plt.show()
