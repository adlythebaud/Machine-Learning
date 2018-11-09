#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:39:42 2018

@author: adlythebaud
"""
from Node import Node
import numpy as np
import random

class MLP():
    def __init__(self, targets = [], num_inputs = 0, layers = 1, nodes_per_layer = 3):
        self.num = 5
        self.targets = targets
        self.num_inputs = num_inputs
        self.nodes = []
        self.layers = layers
        self.nodes_per_layer = nodes_per_layer
        
    def fit(self, X_train, Y_train):
        weights = []
        for i in range(X_train.shape[1]):
            weights.append(random.uniform(-1,1))
        
        for i in np.unique(Y_train):
            self.nodes.append(Node(target_class = i, T = 5))
        
        zipped = zip(X_train, Y_train)
        zipped = list(zipped)
        
        for node in self.nodes:
            # get training data
            training_data = []
            for zip_item in zipped:
                if zip_item[1] == node.target_class:
                    training_data.append(zip_item[0])
            
            # fit data to each node's model
            node.set_weights(weights)
            node.fit(training_data)
            

            
            
    def predict(self, X_test):
        results = []
        for node in self.nodes:
            results.append(node.predict(X_test))

            
            
        return results
        
        
        
        
        
        
        
        
        
        
    
    
        