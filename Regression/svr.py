# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
y = y.reshape(-1,1)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#fitting svr to dataset
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X, y)


y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(y_pred)

print "prediction for 6.5: ", y_pred
#visualizing linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color="blue")
plt.title("Support Vector Regression")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualizing SVR regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("SVR")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
