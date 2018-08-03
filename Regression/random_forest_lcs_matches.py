# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#print full arrays for debugging
np.set_printoptions(threshold=np.nan)

dataset = pd.read_csv('matchinfo.csv')

y = dataset.iloc[:, 5].values
X = dataset.iloc[:, [0,1,2,3,4,7,9,11,13,15,17,19,21,23,25,27]].values

#print y
#print X

#print len(X[0])
#print X[0][15]

#encoding categorical data
#one hot encoder converts the result into binary (0,0,1) -> (0,1,0) instead of 1 and 2
#which is what labelencoder does.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

lengthOfArray = len(X[0])

for i in range(0, lengthOfArray):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    onehotencoder = OneHotEncoder(categorical_features=[i])

X = onehotencoder.fit_transform(X).toarray()
#print X

# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.002, random_state=0)

from sklearn.ensemble import RandomForestRegressor


regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)

#print X_test
print "length of test data", len(X_test)
print list(regressor.predict(X_test))
print y_test
