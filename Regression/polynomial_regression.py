# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:31:19 2017

@author: Omar
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#debugging
print"y: " , y
print"X: " , X

# Fitting for linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


#visualizing linear regression results
plt.scatter(X, y, color ='red')
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Linear regression")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Fitting for polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=10)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualizing polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Polynomial regression")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting new result in lin_reg
print "prediction of linaer regressor", lin_reg.predict(6.5)


#predicting new result in lin_reg_2
print "prediction of poly_regressor",lin_reg_2.predict(poly_reg.fit_transform(6.5))




# Splitting the dataset into the Training set and Test set
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
