#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:05:47 2018

@author: adlythebaud
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv("poker.csv", names = ["suit_1", "rank_1", "suit_2", "rank_2", "suit_3", "rank_3", "suit_4", "rank_4", "suit_5", "rank_5", "hand"])

print(data.shape)

x = data.iloc[:,:10].values
y = data.iloc[:,10:].values

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.33)

