#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:02:08 2018

@author: adlythebaud
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

data = pd.read_csv("newest500.csv")

data = data.drop(['id'], axis=1)
data = data.iloc[:].values

print(int(len(data) / 2))
# shuffle data
indices = random.sample(range(len(data)),int(len(data) * 0.7))

# split into training and test set
train = data[indices]

test = np.delete(data, indices, 0)












