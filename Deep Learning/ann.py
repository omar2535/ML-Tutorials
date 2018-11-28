# Artificial Neural Network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, []]
y = dataset.iloc[:,12].values

X_train, X_test, y_train, y_test = train_test_split(X, y, text_size = 0.25, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

