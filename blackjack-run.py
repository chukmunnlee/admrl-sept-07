import numpy as np
import gym
import random
import pickle

from matplotlib import pyplot as plt
from tqdm import *

EPISODES = 100

capital = 100

winnigs = []

with open('./blackjack_16.pickle', 'rb') as f:
   q_table = pickle.load(f)

normalize_state = lambda st: (st[0], st[1], 1 if st[2] else 0)

def ensure(st, q_table):
   if st not in q_table:
      q_table[st] = np.zeros(2)

def policy(st, q_table):
   # \epsilon greedy policy
   ensure(st, q_table)
   return np.argmax(q_table[st])

env = gym.make('Blackjack-v0')

for _ in range(EPISODES):


   state = normalize_state(env.reset())
   stop = False

   while not stop:
      action = policy(state, q_table)
      new_state, reward, stop, _ = env.step(action)
      if stop:
         if reward == 1:
            capital += 5
         elif reward == -1:
            capital -= 5

         winnigs.append(capital)

      else:
         state = normalize_state(new_state)

   if capital < 0:
      break

plt.plot(winnigs)
plt.title(f'Captial: ${capital}')
plt.show()
