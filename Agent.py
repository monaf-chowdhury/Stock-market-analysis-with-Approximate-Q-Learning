# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 01:56:16 2022

@author: monaf
"""
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from Linear_Model import LinearModel

class Agent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95 # discount factor
        self.epsilon = 1.0 # exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)
        self.q_value = [] # Storing Q values
        
    def act(self,state): # epsilon-greedy! return integer [1,27]
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size) # return random action with probability epsilon
        act_values = self.model.predict(state) 
        self.q_value.append(act_values)
        return np.argmax(act_values[0]) # return the best action based on the Q-values with probability 1-epsilon

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis = 1) # Q-Learning
        
        target_full = self.model.predict(state) # Get the values based on the old parameters W,b
        target_full[0,action] = target # update the entry of the corresponding action

        self.model.sgd(state, target_full) # Stochastic gradient descent. Run one training step and update W, b

        if self.epsilon > self.epsilon_min: #decrease the probability of exploration
            self.epsilon *= self.epsilon_decay

    def load(self,name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
    # No need for modifications as of 6th November, 2022 