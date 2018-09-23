#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 01:48:37 2018

@author: adlythebaud
"""
from HardCodedModel import HardCodedModel

class HardCodedClassifier:
    
    def __init__(self, X_train = [], Y_train = [], X_test = [], Y_test = []):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.model = HardCodedModel()
        
    def fit(self, X_train, Y_train):
        return self.model
            
            
    


    
    
        