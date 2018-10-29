#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:20:35 2018

@author: adlythebaud
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 18:31:08 2018

@author: adlythebaud
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import random


# 1. Get data
iris = datasets.load_iris()

# 2. split data into x and y
x = iris.data
y = iris.target

print(iris.target_names)

# 3. train, test, split    
X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size = .3, random_state = 0)



# 4. feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 5. Initialize weights
weights = []
for i in range(X_train.shape[1]):
    weights.append(random.uniform(-1,1))

# 6. Get outputs for each entry (this is for one node, let's see if it's setosa)
for i in range(X_train.shape[0]):
    tempSum = 0
    for j in range(len(weights)):
        tempSum+= (X_train[i][j] * weights[j])
    
    # get the result for training instance:
    # if y_train[i] = setosa, if tempSum > 0
    

    
    