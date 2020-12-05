import numpy as np
import gym
import random

from matplotlib import pyplot as plt
from tqdm import *

#random.seed(42)
#np.random.seed(42)

ALPHA = 0.02
EPSILON = [ .9, .7, .4, .05, .03 ]

EPISODES = 5000

q_table = np.ones((16, 4))

def policy(st, q_table, eps):
   if random.random() > eps:
      return np.argmax(q_table[st])
   return np.random.choice([0, 1, 2, 3])

def update(st_act, g, q_table, alpha):
   st, act = st_act
   q_table[st][act] += alpha * (g - q_table[st][act])

def epsilon(curr_ep, total_ep, schedule):
   div = total_ep // len(schedule)
   div += 1 if total_ep % len(schedule) > 0 else 0
   return schedule[curr_ep // div ]

env = gym.make('FrozenLake-v0')

wins = []
total_win = 0

for i in tqdm(range(EPISODES)):

   state = env.reset()
   stop = False

   state_action = []
   rewards = []

   while not stop:
      #action = policy(state, q_table, epsilon(i, EPISODES, EPSILON))
      action = policy(state, q_table, .03)
      new_state, reward, stop, _ = env.step(action)

      if stop:
         if reward == 0:
            reward = -10

      state_action.append((state, action))
      rewards.append(reward)

      state = new_state

   if rewards[-1] == 1:
      total_win += 1
   wins.append(total_win/(i + 1))

   visited = set()

   for i in range(len(state_action)):

      if state_action[i] in visited:
         continue

      visited.add(state_action[i])
      gain = sum(rewards[i:])

      update(state_action[i], gain, q_table, ALPHA)

print('wins: ', total_win / EPISODES)
plt.plot(wins)
plt.show()
