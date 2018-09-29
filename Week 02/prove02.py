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

# looping through possible k values.
for i in range(2,20):
    
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
    
    # 5. initialize classifier with chosen k value
    kNN = kNNClassifier(i)
    
    # 6. fit training data and training targets to model
    model = kNN.fit(X_train, Y_train)
    
    # 7. predict, verify accuracy
    countCorrect = 0
    for xt, prediction, yt in zip(X_test, model.predict(X_test), Y_test):
        if prediction == yt:
            countCorrect += 1
            
    # 8. compare to SK Learn classifier
    classifier = KNeighborsClassifier(n_neighbors = 4)
    KNNModel = classifier.fit(X_train, Y_train)
    predictions = KNNModel.predict(X_test)
    
    classifierCountCorrect = 0
    for xt, prediction, yt in zip(X_test, predictions, Y_test):
        if prediction == yt:
            classifierCountCorrect += 1
    # 9. print or visualize results, up to the engineer really
    print("Dev Classifier:     k = %s, Accuracy = %s" % (i, round(countCorrect/len(Y_test) * 100, 2)))
    print("SKLearn Classifier: k = %s, Accuracy = %s" % (i, round(classifierCountCorrect/len(Y_test) * 100, 2)))            
    
        
    
    
    
    
    
    