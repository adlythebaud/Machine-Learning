#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:20:34 2018

@author: adlythebaud

GOAL: Classify a test set of data using K-Nearest Neighbors
INTENTION: Create a K-Nearest Neighbors Classifier and Model,
    classifier will fit data and model will predict.
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets
from kNNClassifier import kNNClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = datasets.load_iris()
x = iris.data
y = iris.target
#y = []
#for t in iris.target:
#    y.append(iris.target_names[t])


X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size = .3, random_state = 42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

kNN = kNNClassifier(4)
model = kNN.fit(X_train, Y_train)



countCorrect = 0
for xt, prediction, yt in zip(X_test, model.predict(X_test), Y_test):
#    print(xt, iris.target_names[prediction], iris.target_names[yt])
    if prediction == yt:
        countCorrect += 1

print("Dev Classifier: Accuracy = %s" % (round(countCorrect/len(Y_test) * 100, 2)))      

for i in range(2,10):
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    #y = []
    #for t in iris.target:
    #    y.append(iris.target_names[t])
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(
            x, y, test_size = .3, random_state = 3)
    
    # Feature Scaling
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    kNN = kNNClassifier(i)
    model = kNN.fit(X_train, Y_train)
    
    
    countCorrect = 0
    for xt, prediction, yt in zip(X_test, model.predict(X_test), Y_test):
    #    print(xt, iris.target_names[prediction], iris.target_names[yt])
        if prediction == yt:
            countCorrect += 1
    classifier = KNeighborsClassifier(n_neighbors = 4)
    KNNModel = classifier.fit(X_train, Y_train)
    predictions = KNNModel.predict(X_test)
    
    classifierCountCorrect = 0
    for xt, prediction, yt in zip(X_test, predictions, Y_test):
    #    print(xt, iris.target_names[prediction], iris.target_names[yt])
        if prediction == yt:
            classifierCountCorrect += 1
    
    print("Dev Classifier:     k = %s, Accuracy = %s" % (i, round(countCorrect/len(Y_test) * 100, 2)))
    print("SKLearn Classifier: k = %s, Accuracy = %s" % (i, round(classifierCountCorrect/len(Y_test) * 100, 2)))            
    
        
    
    
    
    
    
    