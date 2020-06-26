#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:38:35 2020

@author: lenovo
"""

#import libraries and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB

N = 10000
d = 10

ads_selected = []
numbers_of_selections = [0]*d
sums_of_rewards = [0]*d
total_rewards = 0

for n in range(0, 10000):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
                     
        if upper_bound  > max_upper_bound :
            max_upper_bound = upper_bound
            ad = i    
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_rewards += reward
    

#Ploting Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
            
        
    
