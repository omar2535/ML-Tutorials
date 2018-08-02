#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#:-1 means from dataset.iloc[:, 0:exclude last column]
dataset = pd.read_csv('o-ring-erosion-only.data')
X = dataset.iloc[:, [0, 2,3,4]].values
y = dataset.iloc[:, 1].values

print X
print y

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)
print "actual results: ",y_test
print "prediction results: ",y_pred

print "Prediction for temperature of 30deg F:", regressor.predict([[6, 31, 50, 23]])