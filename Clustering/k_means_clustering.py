#K_means clustering
#Unsupervised learning for grouping classes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values
y = dataset.iloc[:, 4].values

#Using elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters= i, init='k-means++', max_iter=300, n_init= 10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("The elbow method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++',max_iter=300, random_state=0)

y_kmeans = kmeans.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='magenta', label='centroids')
plt.title("clusters of clients")
plt.xlabel("Annaul income")
plt.ylabel("Spending Score(1-100)")
plt.show()



# #Actual prediction

# X = dataset.iloc[:, 1:4].values
# y = dataset.iloc[:, 4].values

# #encoding categorical data
# #one hot encoder converts the result into binary (0,0,1) -> (0,1,0) instead of 1 and 2
# #which is what labelencoder does.
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features=[0])
# X = onehotencoder.fit_transform(X).toarray()

# X = X[:, 1:4]
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=0)


