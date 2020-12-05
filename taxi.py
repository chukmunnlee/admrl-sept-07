#import libraries
import numpy as np
import gym
import random
import pickle

from tqdm import *
from matplotlib import pyplot as plt

# tunables/hyperparameters
# learning rate
ALPHA = .02

# randomness
EPSILON=[ .9, .7, .6, .2, .03 ]

# number of episode
EPISODES = 5000

## helper methods
# \epsilon-greedy
def policy(st, eps, q_tabl):
   if random.random() > eps:
      # take the greedy action
      return np.argmax(q_table[st])
   return np.random.choice([0, 1, 2, 3, 4, 5,])

def update(ro, g, q_table):
   st, act = ro
   # new_avg = old_avg + alpha * (gain - old_avg)
   #q_table[st][act] = q_table[st][act] + (ALPHA * (g - q_table[st][act]))
   q_table[st][act] += (ALPHA * (g - q_table[st][act]))

def epsilon(ep, num_ep, schedule):
   num_eps = num_ep // len(schedule)
   num_eps += 1 if num_eps % len(schedule) else 0
   idx = ep // num_eps
   return schedule[idx]

# create the environment
env = gym.make('Taxi-v3')

print('state = ', env.observation_space)
print('action = ', env.action_space)

# stats
success = 0
failed = 0
illegal_dropoff = 0

success_count = []

# create the q-table
q_table = np.ones((500, 6))

for ep in tqdm(range(EPISODES)):
   # reset the environment
   state = env.reset()
   stop = False
   rollout = []
   rewards = []

   # policy evaluation
   while not stop:
      # sample an action from your policy
      # equiproabable
      action = policy(state, epsilon(ep, EPISODES, EPSILON), q_table)

      # act on the environment
      new_state, reward, stop, _ = env.step(action)

      # save the triple of state, action, reward
      rollout.append((state, action))
      rewards.append(reward)

      state = new_state

   if rewards[-1] == -1 or rewards[-1] == -10:
      failed += 1
      success_count.append(0)
   else:
      success += 1
      success_count.append(1)

   # policy improvement 
   # MC first visit
   visited = set()

   for i in range(len(rollout)):

      # check if we have seen (st, act)
      if rollout[i] in visited:
         continue

      visited.add(rollout[i])
      gain = sum(rewards[i:])

      update(rollout[i], gain, q_table)

print('success: ', (success / EPISODES) * 100)
print('failed: ', (failed / EPISODES) * 100)

plt.plot(success_count)
plt.show()

with open('taxi.pickle', 'wb') as f:
   pickle.dump(q_table, f)
