#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:56:03 2020

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
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

#sort
print(resultsinDataFrame.nlargest( n = 10, columns = 'Lift'))

#Output
'''
         Left Hand Side Right Hand Side   Support  Confidence      Lift
3         fromage blanc           honey  0.003333    0.245098  5.164271
0           light cream         chicken  0.004533    0.290598  4.843951
2                 pasta        escalope  0.005866    0.372881  4.700812
8                 pasta          shrimp  0.005066    0.322034  4.506672
7     whole wheat pasta       olive oil  0.007999    0.271493  4.122410
5          tomato sauce     ground beef  0.005333    0.377358  3.840659
1  mushroom cream sauce        escalope  0.005733    0.300699  3.790833
4         herb & pepper     ground beef  0.015998    0.323450  3.291994
6           light cream       olive oil  0.003200    0.205128  3.114710
'''