#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 20:38:09 2018

@author: adlythebaud
"""


import random
import numpy as np
from math import exp

class Node():
    def __init__(self, target = 1, eta = random.uniform(0, 1), T = 0, target_class=""):
        self.target = target
        self.target_class = target_class
        self.eta = eta
        self.T = T
    
    def set_weights(self, weights):
        self.weights = weights
        
    # purpose of this is essentially just weight updating.
    def fit(self, X_train):
        for t in range(self.T):
            for i in range(np.asarray(X_train).shape[0]):
                activation = 0
                
                # compute activation
                for j in range(len(self.weights)):
                    activation+= (X_train[i][j] * self.weights[j])
                
                if activation > 0:
                    activation = 1
                else:
                    activation = 0
                
                # update weights
#                for j in range(len(self.weights)):
#                        self.weights[j] -= self.eta * (activation - self.target) * X_train[i][j]
      
    # this is to test if our weights are correct so we can predict the correct activation.
    def predict(self, X_test):
        results = []
        for i in range(np.asarray(X_test).shape[0]):

            h1 = 0
            
            # compute activation
            for j in range(len(self.weights)):
                h1 += (X_test[i][j] * self.weights[j])
                                 
            # calculate activation
            a1 = 1 / (1 + exp(-h1))
            results.append(a1)
        return results
    

        