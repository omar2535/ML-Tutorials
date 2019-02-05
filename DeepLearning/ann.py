# Artificial Neural Network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])
labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, text_size = 0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# initializing ANN
classifier = Sequential()
# adding first layer, 11 for independant variables
# a good output layer number if to use (#number of independant variables + 1) / 2
classifier.add(Dense(output_dim=6, init= 'uniform', activation = 'relu', input_dim= 11))
# adding second hidden layer
classifier.add(Dense(output_dim=6, init= 'uniform', activation = 'relu'))

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
