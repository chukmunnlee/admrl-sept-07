import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import *
import pickle
import random
import pandas as pd


env=gym.make("CartPole-v0")
random.seed(42)
np.random.seed(42)

ALPHA=0.03
EPSILON=1
N_EPISODES=500
DECAY=0.99
GAMMA=1

#Create weights
weights=np.ones((2,4))
reward_per_ep=[]

tde_final_buffer=[]
tde_working_buffer=[]

#print(env.action_space)
#print(env.observation_space)

def policy(state,epsilon,weights):
    #Shape is (2), from (2,4) dot (4,1)
    dot= np.dot(weights,state)
    if random.random()>epsilon:
        return np.argmax(dot)
    
    return np.random.choice([0,1])

def q_value(state,action,weights):
    dot=np.dot(weights,state)
    if action is None:
        return dot
    return dot[action]
    

for episode in tqdm(range(N_EPISODES)):
    stop=False
    EPSILON *=DECAY
    state=env.reset()
    steps=0
    action=policy(state,EPSILON,weights)
    
    while not stop:
        new_state,reward,stop,_= env.step(action)
        steps+=1
        #env.render()
        
        if stop:
            td_target = reward
            reward_per_ep.append(steps)
            #if steps > 50:
                #for n in 
                #UNpack the working buffer. for i in tde_working etc.
                #np.concatenate((tde_final_buffer,tde_working_buffer))
                
            #tde_working_buffer=[]
            
        else:
            new_action=policy(state,EPSILON,weights)
            max_action=np.argmax(q_value(state,None,weights))
                                 
  #Q learning now! not SARSA. Td_target should be max action, rather than the new action we follow
            td_target=reward + (GAMMA * q_value(new_state,max_action,weights))
            
            #This is SARSA
            td_target=reward + (GAMMA * q_value(new_state,new_action,weights))
            
        # shape is (1,4)
        td_error=  (td_target - q_value(state,action,weights)) * state # state is original gradient mah!
        #print('td_error shape is: ', td_error.shape)
        
        tde_working_buffer.append(td_error)
        
        # shape is (1,4)= 1 * (1,4)
        weights[action] += ALPHA * td_error
            
        state=new_state
        action=new_action
        
df=pd.DataFrame(reward_per_ep)
rolling= df.rolling(5).mean().fillna(0)

plt.plot(reward_per_ep)
plt.plot(rolling)
plt.title("Steps per episode")

plt.show()
