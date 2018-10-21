#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 00:05:18 2018

@author: adlythebaud
"""

import numpy as np

class Node():
    def __init__(self, entropy = 0, column = "", subtree = np.nan):
        self.entropy = entropy
        self.column = column
        self.branches = []
        self.parent = ""
        self.subtree = subtree 
        
        

        
    def __repr__(self):
        return "%s, %s" % (self.column, self.entropy)
    
    
