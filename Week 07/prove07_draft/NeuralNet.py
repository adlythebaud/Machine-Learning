#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:37:26 2018

@author: adlythebaud
"""
from Node import Node
import numpy as np
import random

class NeuralNet():
    def init(self):
        self.nodes = []
        self.layers = []
        
    # adds a layer in between inputs and output(s),
    # with a specified number of nodes in that layer.
    def add_layer(self, num_nodes = 2):    
        # add nodes to layer
        layer = []
        for i in range(num_nodes):
            layer.append(Node())
        self.layers.append(layer)
        # add nodes to hidden layer        
        return layer
    
    
    
    # fit will set the weights and 
    # perform back propagation to update them    
    def fit(self, X_train, Y_train):
        
        # set up training_data
        training_data = zip(X_train, Y_train)                
        
        
        # compute activations for each node in first hidden layer
        """
        [-1]
        []   
        []  [node]
        []  [node]
        []  [node]
        []
        
        multiply the inputs by the node's weights.
        Every hidden layer node has one weight to each input item
        Every input item has one weight to the each hidden layer node
        """
        for i in range(self.layers.shape(0)):
            # construct each node in each layer
            
            
            # set weights for each node in each layer.            
            weights = []
            for node in self.layers[i]:                
                
                # each node will have a set of weights
                node_weights = []
                
                # if we're looking at the first hidden layer,
                # num weights = num input characteristics
                if i == 0:
                    # add a bias node weight
                    for j in range(X_train.shape[1] + 1):
                        node_weights.append(random.uniform(0,1))
                else:
                    print(4)
                node.set_weights(node_weights)
                
                
                
            
#            for node in layer:                
#                # get the activation of each node in this hidden layer
#                a = 0
                
                
        
        
        
        
        
        # update weights
        
        return 0
        
    # predict will return the activation per target class of the input data
    def predict(self, X_test):
        results = []
        return results
    
    def set_weights(self):
        return 0
    
    