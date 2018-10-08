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

# Train Test Split

from sklearn.model_selection import train_test_split, cross_val_predict

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = .3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Use K-Fold Cross Validation, with a KNNClassifier
k = 7
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = k)

cvp = cross_val_predict(classifier, X_train, Y_train, cv = 10)


# Fit and predict data.

KNNModel = classifier.fit(X_train, Y_train)
predictions = KNNModel.predict(X_test)

classifierCountCorrect = 0
for xt, prediction, yt in zip(X_test, predictions, Y_test):
    if prediction == yt:
        classifierCountCorrect += 1
# 9. print or visualize results, up to the engineer really
from sklearn.metrics import f1_score
f1 = f1_score(Y_test, predictions, average = 'micro')
print("Results: k = %s, Accuracy = %s, F1 Score = %s" % (k, round(classifierCountCorrect/len(Y_test) * 100, 2), f1))            







