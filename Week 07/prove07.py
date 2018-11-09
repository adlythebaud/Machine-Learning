#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:13:57 2018

@author: adlythebaud
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from MLP import MLP


# 1. Get data
iris = datasets.load_iris()

# 2. split data into x and y
x = iris.data
y = iris.target

# 3. train, test, split    
X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size = .3, random_state = 0)



# 4. feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 5. Fit and predict
mlp = MLP()
mlp.fit(X_train, Y_train)
for i in X_test:
    print(mlp.predict([i]))

"""
I want to do:
    MLP(targets = targets[], num_inputs = int(num))
    targets are number of classes to identify, 
    num_inputs = number of input nodes on the input layer
    MLP.fit(X_train, Y_train)
    MLP.predict(X_test)
    
    

"""