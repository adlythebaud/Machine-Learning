#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:28:11 2018

@author: adlythebaud
"""

from kNNModel import kNNModel

class kNNClassifier():
    
    def __init__(self, k):

        self.k = k
        

    def fit(self, X_train, Y_train):
        # this should "fit" the data to the model.
        
        # compute distances between all data points: save in an array..
        self.model = kNNModel(X_train, Y_train, self.k)
        return self.model
    
        