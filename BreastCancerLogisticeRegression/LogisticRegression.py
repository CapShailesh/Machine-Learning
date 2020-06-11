#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:35:02 2020

@author: lenovo
"""

#impoerting libraries
import pandas as pd
import numpy as np

#importing dataset
dataset = pd.read_csv('breast_cancer.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1]. values


#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

#training the data
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

from sklearn.svm import SVC
classifier = SVC(random_state=0)
classifier.fit(X_train, y_train)

#predicting the result
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#analyzing the data
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv  =10)
print("Accuracies", accuracies.mean()*100)
print("Standard Deviation", accuracies.std()*100)

'''
Logistic Regression
[[84  3]
 [ 3 47]]
Accuracies 96.70033670033669
Standard Deviation 1.9697976894447813

SVC
[[83  4]
 [ 1 49]]
Accuracies 96.88888888888889
Standard Deviation 2.1699912229119986



DecisionTreeClassifier
[[84  3]
 [ 3 47]]
Accuracies 94.32659932659932
Standard Deviation 2.649823093265441
'''















