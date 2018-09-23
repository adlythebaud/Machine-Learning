#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 00:39:48 2018
Write a python program that can:
    1. Load Data
    2. Prepare training/test sets
    3. Use an existing algorithm to create model 
       (might have to write in separate file)
    4. Use model to create predictions
    5. Implement your own new "algorithm"
    

@author: adlythebaud
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from HardCodedClassifier import HardCodedClassifier


iris = datasets.load_iris()
x = iris.data
y = []
for t in iris.target:
    y.append(iris.target_names[t])


X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, train_size = .7, random_state = 45)

classifier = GaussianNB()
classifier.fit(X_train, Y_train)



print("Predicted --------  Real")
countCorrect = 0
for xt, yt in zip(classifier.predict(X_test), Y_test):
    print(xt, "---------", yt)
    if xt == yt:
        countCorrect += 1        
  

print("Naive Bayes Gaussian Accuracy: ", 100 * round(countCorrect / len(Y_test), 2), "%")

mClassifier = HardCodedClassifier()

model = mClassifier.fit(X_train, Y_train)

predicted = model.predict(X_test)

countCorrect = 0
print("Predicted ------  Real")
for xt, yt in zip(predicted, Y_test):
    if xt == 0:
        xt = "setosa"
    
    print(xt, "---------", yt)
    
    if xt == yt:
        countCorrect += 1
print("HardCodedClassifier Accuracy: ", 100 * round(countCorrect / len(Y_test), 2), "%")

""" EXTRA CREDIT """
#returns a csv
dataset = pd.read_csv("Data.csv")


