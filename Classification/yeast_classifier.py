# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('yeastdata.csv')
X = dataset.iloc[:, 0].values
#y = dataset.iloc[:, 3].values

print X