# import standard libs
import numpy as np
import gym
import random
import pickle

from matplotlib import pyplot as plt
from tqdm import *

random.seed(42)
np.random.seed(42)

# hyperparameters
# learning rate
ALPHA = 0.03
ALPHA = [ .1, .08, .05, .03, .01 ]

STICK_ACTION = 0

# exploration
EPSILON = 0.1
EPSILON=[ .9, .7, .6, .5, .2, .2, .03 ]

# discount
GAMMA = 1

EPISODES = 500000

normalize_state = lambda st: (st[0], st[1], 1 if st[2] else 0)

def decay(ep, num_ep, schedule):
   num_eps = num_ep // len(schedule)
   num_eps += 1 if num_eps % len(schedule) else 0
   idx = ep // num_eps
   return schedule[ min(idx, len(schedule) - 1) ]

def ensure(st, q_table):
   if st not in q_table:
      q_table[st] = np.zeros(2)

def policy(st, q_table, eps):
   # \epsilon greedy policy
   ensure(st, q_table)
   if random.random() > eps:
      # perform the greedy action
      return np.argmax(q_table[st])
   # take the epsilon action
   return np.random.choice([0, 1])

def q_value(st, q_table):
   ensure(st, q_table)
   return q_table[st]

def extract_plot(q_table, action):
   no_ace = []
   has_ace = []
   for p in range(0, 31):
      for d in range(0, 11):
         ensure((p, d, 0), q_table)
         ensure((p, d, 1), q_table)
         if q_table[(p, d, 0)][action] != 0 and action == np.argmax(q_table[(p, d, 0)]):
            no_ace.append((p, d))
         if q_table[(p, d, 1)][action] != 0 and action == np.argmax(q_table[(p, d, 1)]):
            has_ace.append((p, d))
   return (no_ace, has_ace)

# q_table 
q_table = np.ones((30, 11, 2, 2))
q_table = {}

# create the blackjack environment and display the a state
env = gym.make('Blackjack-v0')

wins_per_ep = []
total_wins = 0

for i in tqdm(range(EPISODES)):
   # reset the environment
   state = normalize_state(env.reset())
   action = policy(state, q_table, decay(i, EPISODES, EPSILON))
   reward = 0
   new_action = 0

   stop = False

   while not stop:
      # sample from policy
      # one step
      # print('state = ', state, ', action = ', action, ', reward = ', reward)
      new_state, reward, stop, _ = env.step(action)
      new_state = normalize_state(new_state)

      if STICK_ACTION == action and new_state[0] < 16:
         stop = True
         reward = -1

      # calculate the TD target
      if stop:
         td_target = reward

         if reward == 1:
            total_wins += 1

         wins_per_ep.append(total_wins / (i + 1))

      else:
         new_action = policy(new_state, q_table, decay(i, EPISODES, EPSILON))
         td_target = reward + (GAMMA * q_value(new_state, q_table)[new_action])

      td_error = td_target - q_table[state][action]

      # update the Q(s, a)
      q_table[state][action] += decay(i, EPISODES, ALPHA) * td_error 

      state = new_state
      action = new_action

fig = plt.figure()

# stick
ax = fig.add_subplot(221)
ax.set_title('Stick - 16')
ace, no_ace = extract_plot(q_table, 0)
ax.scatter([x for x, _ in ace ], [ y for _, y in ace ], label='ace')
ax.scatter([x for x, _ in no_ace ], [ y for _, y in no_ace ], label='no ace')
ax.legend()

# hits
ax = fig.add_subplot(222)
ax.set_title('Hits - 16')
ace, no_ace = extract_plot(q_table, 1)
ax.scatter([x for x, _ in ace ], [ y for _, y in ace ], label='ace')
ax.scatter([x for x, _ in no_ace ], [ y for _, y in no_ace ], label='no ace')
ax.legend()

# average wins
ax = fig.add_subplot(223)
ax.set_title('Average wins - 16')
ax.set_title(f'total wins: {((total_wins / EPISODES) * 100)}')
ax.plot(wins_per_ep)

plt.show()

with open('blackjack_16.pickle', 'wb') as f:
   pickle.dump(q_table, f)
