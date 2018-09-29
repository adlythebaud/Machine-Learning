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
        results = []
        
        for x in range(np.shape(X_test)[0]):
            closest = []
            classes = np.zeros(len(np.unique(self.Y_train)))
            
            for i in range(len(self.datapoints)):               
                for j in range(np.shape(X_test)[1]):
                    distance = np.sqrt(np.sum((X_test[x][j] - self.datapoints[i].x[j])**2))
                self.datapoints[i].distance = distance
            self.datapoints.sort(key = lambda x: x.distance)
            
            
            for i in range(self.k):              
               closest.append(self.datapoints[i].y)
           
            # determine most frequent class:
            for i in range(len(closest)):
                classes[closest[i]] += 1  
            
            results.append(np.argmax(classes))
            
                                                  
        return results
    
    
    
    
    
    
    