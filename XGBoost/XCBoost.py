#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:14:48 2020

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))  

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv = 10)
print(accuracies)
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

'''
[[85  2]
 [ 1 49]]
0.9781021897810219
[0.90909091 0.96363636 0.96363636 0.98181818 0.94545455 1.
 0.94545455 0.96296296 1.         0.98113208]
Accuracy : 96.53 %
Standard Deviation: 2.62 %
'''