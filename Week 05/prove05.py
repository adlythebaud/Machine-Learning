#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:05:06 2018

@author: adlythebaud

If all examples have the same label
    return a leaf with that label
Else if there are no features left to test
    return a leaf with the most common label
Else
    Consider each available feature
    Choose the one that maximizes information gain
    Create a new node for that feature

    For each possible value of the feature
        Create a branch for this value
        Create a subset of the examples for each branch
        Recursively call the function to create a new node at that branch        
"""
from sklearn import datasets
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import mode

def calc_entropy(p):
    if p != 0:
        return (-1 * p * np.log2(p))
    else:
        return 0

# 1. Get system entropy
# Get a test dataset.
df = pd.read_csv('voting_data.csv', na_values = '?', 
                      names = ['class_name', 'handicapped_infants',
                               'water_project_cost_sharing', 'adoption_of_the_budget_resolution',
                               'physician-fee-freeze', 'el-salvador-aid', 
                               'religious-groups-in-schools',
                               'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                               'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                               'education-spending', 'superfund-right-to-sue', 'crime',
                               'duty-free-exports', 'export-administration-act-south-africa'])

df = df.applymap(str)
values = {}
for column in df:
    values.update({column: mode(df[column], nan_policy = 'omit').mode[0]})
df = df.fillna(value = values)
# split data into x and y
x = df.iloc[:,1:].values
y = df.iloc[:,:1].values


print(y)    
#print(Counter(y).keys())
#print(Counter(y).values())

#system_entropy = 0
#for i in Counter(y).values():
#    system_entropy += calc_entropy(i / len(y))
#    
## 2. Calculate information gain to determine root node.
#for i in x:
#    print(i)

