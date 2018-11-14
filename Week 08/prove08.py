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
# so many for loops.....

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

# 3. train, test, split    
X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size = .3, random_state = 0)

# 4. feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4.5 split training data based on target class, 
# add as an element -1 to each input
zipped = zip(X_train, Y_train)
training_group = []
for i in np.unique(Y_train):
    training_group.append([])


for datapoint in zipped:
    datapoint = np.asarray(datapoint)
    datapoint = datapoint.tolist()
    datapoint[0] = datapoint[0].tolist()
    datapoint[0].append(-1)
    datapoint[0] = np.asarray(datapoint[0])
    training_group[datapoint[1]].append(np.asarray(datapoint))



# 5. Decide how many layers and how many nodes per layer.

# two layers (one hidden and one output, with 3 nodes in hidden layer)
# this works, as long as you don't throw in a random layer with only one node in the middle.
layers = [3,1]


# 6. Create weights for each node in each layer
weights = []

for layer in layers:    
    # loop through each node, add a weight for every input from previous layer
    layer_weights = []
    for i in range(layer):
        node_weights = []        
        # this for loop only needs to happen once. 


        # if layer.index == 0.... 
        if layers.index(layer) == 0:
            for j in range(X_train.shape[1] + 1):
                node_weights.append(random.uniform(-1,1))
        else:
        # make weights for every node in PREVIOUS layer
            for j in range(layers[layers.index(layer) - 1] + 1):
                node_weights.append(random.uniform(-1,1))
            
            
        layer_weights.append(node_weights)
    weights.append(layer_weights)

# 7. Helper functions and initializations:
def activation(num):
    return 1 / (1 + np.exp(-num))

### ERROR FUNCTION ###
def error(layers, layer, result, target, weights = [], error = 0):
    if (layers.index(layer) == (len(layers) - 1)):
        # computing error of output layer
        print("computing output layer error")
        return result * (1 - result) * (result - target)
    else:
        print("computing hidden layer error")
        return result * (1 - result) * np.dot(weights, error)
    return 0

eta = random.uniform(0,1)   
     
# 8. set up feed forward
def output(layer, weights, input, bias = False):
    a = []
    for i in range(layer):
        a.append(activation(np.dot(input, weights[layers.index(layer)][i])))
    if bias == True:
        a.append(-1)
    return a
    

def compute(layers, weights, input):   
    layer_results = []
    actual_results = []
    for layer in layers:        
        if layers.index(layer) != (len(layers) - 1):            
            # We're on a hidden layer
            # make sure output is a list.
            layer_results.append(output(layer, weights, input, True))

        else:
            # we're on output layer                    
            actual_results.append(output(layer, weights, layer_results, False))
            
    return [layer_results[0], actual_results[0][0][0]]
  
#GOAL: Compute activation depending on the layer we're on.      

# create a neural network for each target class
# you'll need to train the weights with ALL inputs for each neural net
# try making the nn and updating weights for classifying 
# whether something is setosa (0) or not


# 9. Update weights   
results = compute(layers, weights, training_group[0][0][0])

print(error(layers, layers[len(layers) - 1], results[1], 1))








        
        