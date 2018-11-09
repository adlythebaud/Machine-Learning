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
from Node import Node


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

# 5. Initialize weights
weights = []
for i in range(X_train.shape[1]):
    weights.append(random.uniform(-1,1))
    
# 6. Initialize our learning rate, eta
eta = 0.0453

# 7. TRAIN. 
# create n nodes for n target classes:
nodes = []
for i in np.unique(Y_train):
    nodes.append(Node(target_class = i, T = 5))

zipped = zip(X_train, Y_train)
zipped = list(zipped)

#for node in nodes:
#    # get training data
#    training_data = []
#    for zip_item in zipped:
#        if zip_item[1] == node.target_class:
#            training_data.append(zip_item[0])
#    
#    # fit data to each node's model
#    node.set_weights(weights)
#    node.fit(training_data)
#    print(node.predict([training_data[0]]))

# get training data
training_class = 0
training_data = []
for zip_item in zipped:
    if zip_item[1] == nodes[training_class].target_class:
        training_data.append(zip_item[0])

# fit data to each node's model
nodes[training_class].set_weights(weights)
nodes[training_class].fit(training_data)
print(nodes[training_class].predict([training_data[0]]))

# loop through arbitary number of hidden layer nodes:
for i in range(3):
    print(nodes[training_class].predict([training_data[0]]))


# len = num attributes of x
input_nodes = []

hidden_nodes = []

# one output node for yes or no on whether we hit our target class.
# one output node per target class.
output_node = Node()


    
    
    
    
    
    