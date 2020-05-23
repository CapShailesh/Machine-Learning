#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:52:23 2020

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv");

X = dataset.iloc[:, 1:2].values;
y = dataset.iloc[:, 2].values;


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, y)


plt.scatter(X, y, color = "red");
plt.plot(X, regressor.predict(X))
plt.show()



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=9)

X_poly = poly_reg.fit_transform(X, y);

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y);

plt.scatter(X, y, color = "red");
plt.plot(X, lin_reg2.predict(X_poly))
plt.show()

regressor.predict(6.5)
lin_reg2.predict(poly_reg.fit_transform(6.5))

