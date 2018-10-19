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
from scipy.stats import mode, itemfreq

def entropy(p):
    if p != 0:
        return (-1 * p * np.log2(p))
    else:
        return 0

# create function to calculate a column's entropy
def column_entropy(column):
    sum_entropy = 0
    for i in np.unique(column, return_counts = True)[1]:
        sum_entropy += entropy(i / len(column))
    return sum_entropy




# 1. Get system entropy
# Get a test dataset.
df = pd.read_csv('voting_data.csv', 
                      names = ['class_name', 'handicapped_infants',
                               'water_project_cost_sharing', 'adoption_of_the_budget_resolution',
                               'physician-fee-freeze', 'el-salvador-aid', 
                               'religious-groups-in-schools',
                               'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                               'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                               'education-spending', 'superfund-right-to-sue', 'crime',
                               'duty-free-exports', 'export-administration-act-south-africa'])

df = df.applymap(str)

# df.replace({'column-to-check': {'value-to-look-for': 'replacement'}})
for column in df:
    df = df.replace({column: {'?': mode(df[column], nan_policy = 'omit').mode[0]}})
 
# split data into x and y
x = df.iloc[:,1:].values
y = df.iloc[:,:1].values


e_start = column_entropy(df.iloc[:,:1])

# 2. Calculate information gain to determine root node.
#   calculate entropy of all columns
for i in df.columns:
    if i == df.iloc[:,:1].columns[0]:
        continue
    else:
        # CALCULATE ENTROPY OF THE COLUMNS
        
        # branches are unique values of the column
        
        # get unique values of column to iterate over.
        for j in np.unique(df[i].values):
            
            # get the rows that are equal to the branch (unique value)
            rows = df.loc[df[i] == j]
        
            # calculate the entropy of the target class among those rows.
            print(i, j, column_entropy(rows.iloc[:,:1]))



