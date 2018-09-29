#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 02:25:37 2018

@author: adlythebaud
"""

class DataPoint():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distance_from_test_point = 0
        
    def __repr__(self):
        return "%s, %s" % (self.x, self.y)