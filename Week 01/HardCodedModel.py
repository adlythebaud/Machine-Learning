#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 13:38:51 2018

@author: adlythebaud
"""

class HardCodedModel:
    def __init__(self):
        self.targets_predicted = []
        
    def predict(self, X_test):
        # return an array of size X_test,
        # all zeroes or Iris-Setosa
        for i in X_test:
            self.targets_predicted.append(0)
        return self.targets_predicted