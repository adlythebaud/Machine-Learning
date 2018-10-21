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

import numpy as np
import pandas as pd
from Node import Node
from scipy.stats import mode
from operator import attrgetter

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
df_copy = df

def info_gain(dframe):
    entropies = []
    for i in dframe.columns:
        
        # CALCULATE ENTROPY OF THE COLUMNS        
        # get unique values of column to iterate over.
        a = []
        for j in np.unique(df[i].values):
            
            # get the rows that are equal to the branch (unique value)
            rows = dframe.loc[df[i] == j]
            
            # calculate the entropy of the target class among those rows (of the branch)
            # get weighted average of entropies among branches of column, add to entropies list
            a.append((len(rows) / len(dframe[i].values)) * (column_entropy(rows.iloc[:,:1])))
            
            entropies.append(Node(entropy = np.sum(a), column = i))
    # this column has the lowest entropy, so it will be our root node.
    root = min(entropies, key = attrgetter('entropy'))
        
    # return the node.
    return root

print(info_gain(df))


# now to get child nodes....

def make_tree(df):
    # if all examples have the same label, return leaf with that label.
    if len(np.unique(df.iloc[:,:1].values)) == 1:
        return np.unique(df.iloc[:,:1].values)
    # if there are no features left, return leaf with most common label among data.
    elif len(df.columns) == 0:
        return mode(df.iloc[:].values, nan_policy = 'omit').mode[0]
    else:
        # calculate info gain and create node for that feature.        
        node = info_gain(df) #where do I set it's parent??!?
        
        # create branches/edges for possible values of new Node
        node.branches = np.unique(df[node.column].values)
            
        # create a subset of examples that have this value    
        for i in node.branches:
            df = df.loc[df[node.column] == i]
            df = df.drop([node.column], axis = 1)
            make_tree(df)
make_tree(df)




        
                         
        



    





