#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:55:57 2018

@author: adlythebaud
"""

import pandas as pd
import numpy as np

df = pd.read_csv("car_data.csv", 
                 names = ["buying", "maintenance", "doors", 
                          "persons", "lug_boot", "safety", "class"])

#print(df.iloc[:,:6].values)

# Encode...
# These for loops don't account for messy data.
for i in df.iloc[:,2].values:
    if i == '5more':
        i = 5
    else:
        i = int(i)

for i in df.iloc[:,3].values:
    if i == 'more':
        i = 5
    else:
        i = int(i)

x = df.iloc[:,:6].values
y = df.iloc[:,6:].values


# Encode Categorical Data
#convert y to 1-dimensional array using ravel.
y = y.ravel()

# Label encode your data, one column at a time.
from sklearn.preprocessing import LabelEncoder


for i in range (6):
    trainLE = LabelEncoder()
    x[:,i] = trainLE.fit_transform(x[:,i])


le = LabelEncoder()    
y = le.fit_transform(y)








