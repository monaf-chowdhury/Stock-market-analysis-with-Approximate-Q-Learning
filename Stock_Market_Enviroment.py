# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 04:07:01 2022

@author: monaf
"""
import numpy as np 
import pandas as pd 
import itertools
from datetime import datetime 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

class StockEnv:
    '''
    Enviroment with Number_of_stocks total stocks
    Actions: sell, hold, buy the ith stock, for i in [1,Number_of_stocks], 3^i total actions

    This project considers three stocks (Number_of_stocks): Beximco, Grameenphone, Square
    '''
    def __init__(self,data,initial_investment,num_stocks=3):

        #data
        self.stock_price_history = data
        self.num_samples, self.num_features =  self.stock_price_history.shape
        self.num_stocks = num_stocks
        
        #attributes
        self.initial_investment=initial_investment
        self.curr_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.stock_features = None
        
        self.action_space = np.arange(3**self.num_stocks)                       # 3 actions sell, hold, buy
        self.action_list = list(map(list, itertools.product(np.arange(self.num_stocks), repeat=self.num_stocks)))

        self.state_dimension = self.num_features + self.num_stocks + 1
        self.prices = list(self.stock_price_history.columns)[::6]               # prices of all three stocks 
        self.stock_columns = list(self.stock_price_history.columns)[0:]         # lists all columns 
        
        self.reset()

    def reset(self):
        self.curr_step = 0
        self.stock_owned = np.zeros(self.num_stocks)
        self.cash_in_hand = self.initial_investment
        self.stock_price = self.stock_price_history.loc[self.curr_step,self.prices] # This will take prices of stock at each step 
             
        self.stock_features = self.stock_price_history.loc[self.curr_step,self.stock_columns] # This will contain all the values of features at each step
   
        return self.get_state()

    def step(self,action):
        assert action in self.action_space                 # return assertion error if there is no such an action
        previous_value = self.get_value()                   # get the value before the action

        self.stock_price = self.stock_price_history.loc[self.curr_step,self.prices] # update prices using the next sample
        self.stock_features = self.stock_price_history.loc[self.curr_step,self.stock_columns]

        
        #perform action 
        self.trade(action) 
    
        self.curr_step +=1 # go to the next day
        
        self.stock_price = self.stock_price_history.loc[self.curr_step,self.prices] # update prices using the next sample
        self.stock_features = self.stock_price_history.loc[self.curr_step,self.stock_columns]

        curr_value=self.get_value() # get the new value
        
        # find the reward
        reward = curr_value - previous_value
        # End of the data
        done = (self.curr_step == self.num_samples-1)
        info = {'current value': curr_value}

        return self.get_state(), reward, done, info
           
    
    def get_state(self):
        state = np.empty(self.state_dimension)
        state[0:self.num_features] = self.stock_features
        state[self.num_features : self.num_features + self.num_stocks ] = self.stock_owned 
        state[-1] = self.cash_in_hand
        
        return state

    def get_value(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand 

    def trade(self,action):

        # example of action for 3 stock: action = [1, 0 ,2]
        # 0: sell
        # 1: hold
        # 2: buy
        action_vector = self.action_list[action]
        sell_index = []
        buy_index = []
        
        for i in range(len(action_vector)):
            if action_vector[i]==0:
                sell_index.append(i)
            elif action_vector[i]==2:
                buy_index.append(i)
        
            
        # first sell
        if sell_index:
            for i in sell_index:
                self.cash_in_hand = self.cash_in_hand + self.stock_owned[i] * self.stock_price[i]
                self.stock_owned[i] = 0 #sell them all
         
        if buy_index:
            can_buy = True  
            while can_buy:
                can_buy = False
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.cash_in_hand = self.cash_in_hand - self.stock_price[i]
                        self.stock_owned[i] += 1

                        if self.cash_in_hand > self.stock_price[i]:
                            can_buy = True
         
    
###### No need for modifications as of 10th November, 2022 #############################
   
                        