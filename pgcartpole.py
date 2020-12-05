import numpy as np
import gym
import pandas as pd

from tqdm import *
from matplotlib import pyplot as plt

ALPHA = 0.001
GAMMA = 1

EPISODES = 800

weights = np.ones((4))

def sigmoid(st, w):
   dot = np.dot(st, w)
   return 1 / (1 + np.exp(-dot))

def act_prob(st, w):
   p0 = sigmoid(st, w)
   return (p0, 1 - p0)

def policy(st, w):
   probs = act_prob(st, w)
   return np.random.choice([0, 1], p=act_prob(st, w))

def policy_grad(st, w):
   sig = sigmoid(st, w)
   grad_p0 = st - (st * sig)
   grad_p1 = - (st * sig)

   return (grad_p0, grad_p1)

def gains(rewards, gamma):
   g = []
   total = 0
   for i in reversed(range(len(rewards))):
      total = rewards[i] + (gamma * total)
      g.insert(0, total)
   return np.array(g)

env = gym.make('CartPole-v0')

reward_per_ep = []

for i in tqdm(range(EPISODES)):

   stop = False
   state = env.reset()

   state_action = []
   rewards = []

   while not stop:
      action = policy(state, weights)
      new_state, reward, stop, _ = env.step(action)

      state_action.append((state, action))
      rewards.append(reward)

      state = new_state

   g = gains(rewards, GAMMA)

   reward_per_ep.append(g[0])

   grad_act = np.array([ policy_grad(st, weights)[act] for st, act in state_action ])
   delta = np.dot(g, grad_act)

   weights += ALPHA * delta

plt.plot(reward_per_ep)

df = pd.DataFrame(reward_per_ep)
rolling = df.rolling(5).mean().fillna(0)
plt.plot(rolling, c='r')

plt.show()
