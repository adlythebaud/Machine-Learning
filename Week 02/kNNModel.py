#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:38:24 2018

@author: adlythebaud
"""
import numpy as np
from DataPoint import DataPoint

class kNNModel():
    
    def __init__(self, X_train, Y_train, k):
        self.X_train = X_train
        self.Y_train = Y_train
        self.k = k
        self.datapoints = []
        for xt, yt in zip(self.X_train, self.Y_train):
            self.datapoints.append(DataPoint(xt, yt))

        
        
    def predict(self, X_test):
        # Return an array of correct answers based off these test inputs (X_test)
        # compute distance between X_test[i] and points
        # get distances between X_train and X_test:
        closest = []
        results = []
        for x in range(np.shape(X_test)[0]):
            
            
            for i in range(len(self.datapoints)):
                #you could make this code better by not hardcoding columns
                distance = np.sqrt(
                        (X_test[x][0] - self.datapoints[i].x[0])**2 + 
                        (X_test[x][1] - self.datapoints[i].x[1])**2 +
                        (X_test[x][2] - self.datapoints[i].x[2])**2 +
                        (X_test[x][3] - self.datapoints[i].x[3])**2)
                
                self.datapoints[i].distance = distance
            self.datapoints.sort(key = lambda x: x.distance)
            
            
            for i in range(self.k):
               closest.append(self.datapoints[i].y)
           
            # determine most frequent class:                                                                   
        return closest