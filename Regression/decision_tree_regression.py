#Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#:-1 means from dataset.iloc[:, 0:exclude last column]
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)


y_pred = regressor.predict(6.5)

#must use higher resolution as decision tree takes average of values at 2 points as 1 interval,
#so when a point lies in the interval, the value will = the average of the values at the 2 points

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid), color="blue")
plt.title("Decision tree regression")
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

print "prediction for level=6.5", y_pred