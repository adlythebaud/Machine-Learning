#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 00:05:18 2018

@author: adlythebaud
"""

class Node():
    def __init__(self, entropy = 0, column = ""):
        self.entropy = entropy
        self.column = column
        self.children = []
        self.parent = ""
        
        

        
    def __repr__(self):
        return "%s, %s" % (self.column, self.entropy)
    
    
