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
        
        # 1. loop through all test inputs
        for x in range(np.shape(X_test)[0]):
            
            # 2. array of closest neighbors
            closest = []
            
            # 3. array of different classes and their frequencies, each index is a unique class.
            classes = np.zeros(len(np.unique(self.Y_train)))            
            
            # 4. get distance from test input to each item in training set
            for i in range(len(self.datapoints)):               
                
                # 5. compute euclidean distance; this loops through all dimensions of training set
                for j in range(np.shape(X_test)[1]):
                    distance = np.sqrt(np.sum((X_test[x][j] - self.datapoints[i].x[j])**2))
                
                # 6. set the distance from test point for each item in training set
                self.datapoints[i].distance_from_test_point = distance
            
            # 7. sort the datapoints in training set, so that nearest neighbors are on top
            self.datapoints.sort(key = lambda x: x.distance_from_test_point)
            
            # 8. get the k-nearest neighbors off the top of sorted training set
            for i in range(self.k):              
               closest.append(self.datapoints[i].y)
           
            # 9. determine most frequent class:
            for i in range(len(closest)):
                classes[closest[i]] += 1  
            
            # 10. insert result of most frequent class into list
            results.append(np.argmax(classes))
            
        # 11. return results
        return results
    
    
    
    
    
    
    