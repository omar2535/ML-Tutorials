# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#print full arrays for debugging
np.set_printoptions(threshold=np.nan)

dataset = pd.read_csv('matchinfo.csv')

y = dataset.iloc[:, 5].values
X = dataset.iloc[:, [0, 1, 2, 3, 4, 7, 9, 11,
                     13, 15, 17, 19, 21, 23, 25, 27]].values


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


# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=0)

# Feature Scaling (standardize the values)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Making the classifier(Kernel SVM)
from sklearn.svm import SVC

classifier = SVC(kernel= 'rbf', random_state=0)
classifier.fit(X_train, y_train)
'''
RESULTS:
    POLY deg=3 GAVE [[7 29], [7 34]]
    RBF GAVE        [[20 16], [17 24]]
    SIGMOID GAVE    [[19 17], [25 16]]
'''


#predict 
y_pred = classifier.predict(X_test)

#making the confusion matrix(to evalute our model accuracy)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print cm

