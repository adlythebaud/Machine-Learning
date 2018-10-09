#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:55:57 2018

@author: adlythebaud
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy.stats import mode

""" 
Car Data
"""
def car_data():
    
    # 1. Read in CSV
    df = pd.read_csv("car_data.csv", 
                     names = ["buying", "maintenance", "doors", 
                              "persons", "lug_boot", "safety", "class"])        
    
    # 2. Split to x and y
    x = df.iloc[:,:6].values
    y = df.iloc[:,6:].values
    
    
    # 3. Encode Categorical Data
    
    # convert y to 1-dimensional array using ravel.
    y = y.ravel()
    
    # Label encode one column at a time.
    for i in range (6):
        trainLE = LabelEncoder()
        x[:,i] = trainLE.fit_transform(x[:,i])
    
    le = LabelEncoder()    
    y = le.fit_transform(y)
    
    # 4. Train Test Split
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = .3)
    
    # 5. Feature Scaling
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    # 6. Use K-Fold Cross Validation, with a KNNClassifier
    k = 7    
    classifier = KNeighborsClassifier(n_neighbors = k)
    cvp = cross_val_predict(classifier, X_train, Y_train, cv = 10)
    
    # 7. Fit and predict data.
    
    KNNModel = classifier.fit(X_train, Y_train)
    predictions = KNNModel.predict(X_test)
    
    classifierCountCorrect = 0
    for xt, prediction, yt in zip(X_test, predictions, Y_test):
        if prediction == yt:
            classifierCountCorrect += 1
    
    # 8. print or visualize results
    from sklearn.metrics import f1_score
    f1 = f1_score(Y_test, predictions, average = 'micro')
    print("Results: k = %s, Accuracy = %s, F1 Score = %s" % (k, round(classifierCountCorrect/len(Y_test) * 100, 2), f1))            


""" 
Autism Data
"""
def autism_data():    
    
    # 1. Read in CSV
    df = pd.read_csv("adult_autism_data.csv", na_values = '?')
    df = df.applymap(str)    
    values = {}
    
    # 2. Fill in missing values
    for column in df:
        values.update({column: mode(df[column], nan_policy = 'omit').mode[0]})            
    df = df.fillna(value = values)
    
    # 3. Split to X and Y
    x_end = len(df.columns) - 1    
    x = df.iloc[:,1:x_end].values
    y = df.iloc[:,x_end:].values


    # 4. Encode Categorical Data
    
    #convert y to 1-dimensional array using ravel.
    y = y.ravel()
    
    # Label encode one column at a time
  
    for i in range(0,x.shape[1]):
        trainLE = LabelEncoder()
        x[:,i] = trainLE.fit_transform(x[:,i])
 
    le = LabelEncoder()    
    y = le.fit_transform(y)
    
    # 5. Train Test Split
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = .3)
    
    # 6. Feature Scaling
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    # 7. Use K-Fold Cross Validation, with a KNNClassifier
    k = 7    
    classifier = KNeighborsClassifier(n_neighbors = k)
    cvp = cross_val_predict(classifier, X_train, Y_train, cv = 10)
    
    # 8. Fit and predict data.
    
    KNNModel = classifier.fit(X_train, Y_train)
    predictions = KNNModel.predict(X_test)
    
    classifierCountCorrect = 0
    for xt, prediction, yt in zip(X_test, predictions, Y_test):
        if prediction == yt:
            classifierCountCorrect += 1
    
    # 9. print or visualize results
    from sklearn.metrics import f1_score
    f1 = f1_score(Y_test, predictions, average = 'micro')
    print("Results: k = %s, Accuracy = %s, F1 Score = %s" % (k, round(classifierCountCorrect/len(Y_test) * 100, 2), f1))            

def mpg():

    # 1. Read in data as a fixed width file    
    df = pd.read_fwf("mpg.txt", sep = " ", na_values = '?', 
                     names = ['mpg', 
                              'cylinders', 
                              'displacement', 
                              'horsepower', 
                              'weight', 
                              'acceleration', 
                              'model_year', 
                              'origin', 
                              'car_name'])
    
    # 2. Get rid of quotation marks
    df['mpg'] = df['mpg'].str.replace('"','')
    df['car_name'] = df['car_name'].str.replace('"','')
    df = df.drop(['car_name'], axis = 1)

    # 3. Replace missing values
    values = {}
    for column in df:
        values.update({column: mode(df[column], nan_policy = 'omit').mode[0]})            
    df = df.fillna(value = values)
    
    # 4. Split into X and Y    
    x_end = len(df.columns) - 1    
    x = df.iloc[:,1:x_end].values
    y = df.iloc[:,0].values
    
    # 5. Encode Categorical Data
    
    # convert y to 1-dimensional array using ravel.
    y = y.ravel()
    
    # Label encode one column at a time.
    for i in range(0,x.shape[1]):
        trainLE = LabelEncoder()
        x[:,i] = trainLE.fit_transform(x[:,i])
 
    le = LabelEncoder()    
    y = le.fit_transform(y)
    
    # 6. Train Test Split
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = .3)
    
    # 7. Feature Scaling
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
        
    # 8. Use K-Fold Cross Validation, with a KNNRegressor
    k = 7   
    regressor = KNeighborsRegressor(n_neighbors = k)
    cvp = cross_val_predict(regressor, X_train, Y_train, cv = 10)
    
    # 9. Fit and predict data.
    
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    
    classifierCountCorrect = 0
    for xt, prediction, yt in zip(X_test, predictions, Y_test):
        if prediction == yt:
            classifierCountCorrect += 1
    
    # 10. print or visualize results
    

    print("Results: k = %s, Accuracy = %s" % (k, round(classifierCountCorrect/len(Y_test) * 100, 2)))            

def main():
    car_data()
    autism_data()
    mpg()

main()


