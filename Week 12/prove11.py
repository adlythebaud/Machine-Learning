#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:05:47 2018

@author: adlythebaud
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import datasets
from scipy.stats import mode


############################################
# POKER DATA
data = pd.read_csv("poker.csv", names = ["suit_1", "rank_1", "suit_2", "rank_2", "suit_3", "rank_3", "suit_4", "rank_4", "suit_5", "rank_5", "hand"])
test_data = pd.read_csv("poker_test.csv", names = ["suit_1", "rank_1", "suit_2", "rank_2", "suit_3", "rank_3", "suit_4", "rank_4", "suit_5", "rank_5", "hand"])
data.append(test_data)
x = data.iloc[:,:10].values
y = data.iloc[:,10:].values
############################################

############################################
# IRIS DATA
#iris = datasets.load_iris()
#x = iris.data
#y = iris.target
############################################


############################################
# BREAST CANCER DATA
#data = pd.read_csv("breast_cancer.csv", na_values = "?")
#values = {}
#for column in data:
#    values.update({column: mode(data[column], nan_policy = 'omit').mode[0]})            
#data = data.fillna(value = values)
#x = data.iloc[:,:9].values
#y = data.iloc[:,10:].values
############################################
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.33)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# 1. DECISION TREE CLASSIFIER
dtree = DecisionTreeClassifier(max_depth = 5, criterion = "entropy")
dtree.fit(X_train, Y_train)

print("Decision Tree Score: ", dtree.score(X_test,Y_test))

Y_train = np.ravel(Y_train)

# 2. MLP CLASSIFIER
mlp = MLPClassifier(max_iter = 300)

mlp.fit(X_train, np.ravel(Y_train))
print("MLP Score: ", mlp.score(X_test, Y_test))

# 3. KNN CLASSIFIER
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
print("KNN Score: ", knn.score(X_test, Y_test))

# 4. BAGGING CLASSIFIER
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples = 1.0)
bagging.fit(X_train, Y_train)
print("Bagging Score: ", bagging.score(X_test, Y_test))

# 5. RANDOM FOREST CLASSIFIER
random_forest = RandomForestClassifier(n_estimators = 10)
random_forest.fit(X_train, Y_train)
print("Random Forest Score: ", random_forest.score(X_test, Y_test))

# 6. ADABOOST CLASSIFIER
ada = AdaBoostClassifier(n_estimators = 50)
ada.fit(X_train, Y_train)
print("AdaBoost Score: ", ada.score(X_test, Y_test))

# 7. GRADIENT BOOSTING CLASSIFIER
gr_booster = GradientBoostingClassifier(n_estimators = 50).fit(X_train, Y_train)
print("GradientBoosting Score: ", gr_booster.score(X_test, Y_test))






