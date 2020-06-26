#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:06:12 2020

@author: lenovo
"""
#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset and preprocess
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])    
    
   
#training the Apriori model
from apyori import apriori
rules = apriori(transactions = transactions, min_support = .003, min_confidence = 0.2, min_lift=3, min_length = 2, max_length = 2)

#visualize the result
results = list(rules)
print(results)


#organizing the result
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

#sort
print(resultsinDataFrame.nlargest( n = 10, columns = 'Support'))

"""
              Product 1    Product 2   Support
4         herb & pepper  ground beef  0.015998
7     whole wheat pasta    olive oil  0.007999
2                 pasta     escalope  0.005866
1  mushroom cream sauce     escalope  0.005733
5          tomato sauce  ground beef  0.005333
8                 pasta       shrimp  0.005066
0           light cream      chicken  0.004533
3         fromage blanc        honey  0.003333
6           light cream    olive oil  0.003200
"""
