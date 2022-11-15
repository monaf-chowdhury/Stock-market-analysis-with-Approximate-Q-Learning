# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 04:05:04 2022

@author: monaf
"""
import numpy as np 
import pandas as pd 


class LinearModel:
    # Linear Regression
    def __init__(self,state_dim,action_dim):
        self.W = np.random.randn(state_dim,action_dim) / np.sqrt(state_dim)
        self.b = np.zeros(action_dim)

        # momentum terms 
        self.vW = 0
        self.vb = 0

        self.losses = []
        self.prediction = []
        self.actual_value = [] 
        

    def predict(self,X):
        assert(len(X.shape)==2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9): # Stochastic gradent descent
        assert(len(X.shape)==2)

        num_values=np.prod(Y.shape)

        Yhat= self.predict(X)
        gW= 2*X.T.dot(Yhat - Y)/num_values
        gb= 2*(Yhat-Y).sum(axis=0)/num_values

        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb 

        self.W +=  self.vW
        self.b +=  self.vb

        mse = np.mean((Yhat-Y)**2)
        self.losses.append(mse)

    def save_weights(self,filepath):
        np.savez(filepath, W=self.W, b=self.b)

    def load_weights(self,filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']
        
### No need for further modification i think as of 4th November, 2022