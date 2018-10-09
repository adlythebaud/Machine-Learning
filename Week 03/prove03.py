#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:55:57 2018

@author: adlythebaud
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode

""" 
Car Data
"""
def car_data():
    df = pd.read_csv("car_data.csv", 
                     names = ["buying", "maintenance", "doors", 
                              "persons", "lug_boot", "safety", "class"])        
    
    x = df.iloc[:,:6].values
    y = df.iloc[:,6:].values
    
    
    # Encode Categorical Data
    
    #convert y to 1-dimensional array using ravel.
    y = y.ravel()
    
    # Label encode your data, one column at a time.
    
    
    for i in range (6):
        trainLE = LabelEncoder()
        x[:,i] = trainLE.fit_transform(x[:,i])
    
    le = LabelEncoder()    
    y = le.fit_transform(y)
    
    # Train Test Split
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = .3)
    
    # Feature Scaling
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    # Use K-Fold Cross Validation, with a KNNClassifier
    k = 7    
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


""" 
Autism Data
"""
def autism_data():    
    df = pd.read_csv("adult_autism_data.csv", na_values = '?')
    df = df.applymap(str)
    # Encode
    # Set np.nan values = "NaN" or 0, depending on column data type    
    values = {}
    for column in df:
        values.update({column: mode(df[column]).mode[0]})        

    
    df = df.fillna(value = values)
    
    x_end = len(df.columns) - 1    
    x = df.iloc[:,1:x_end].values
    y = df.iloc[:,x_end:].values


    # Encode Categorical Data
    
    #convert y to 1-dimensional array using ravel.
    y = y.ravel()
    
    # Label encode your data, one column at a time.
  
    for i in range(0,x.shape[1]):
        trainLE = LabelEncoder()
        x[:,i] = trainLE.fit_transform(x[:,i])
 
    le = LabelEncoder()    
    y = le.fit_transform(y)
    
    # Train Test Split
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = .3)
    
    # Feature Scaling
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    # Use K-Fold Cross Validation, with a KNNClassifier
    k = 7    
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




def main():
    car_data()
    autism_data()

main()


