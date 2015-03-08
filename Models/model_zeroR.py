__author__ = 'jrbaker'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import numpyArrays
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from zeroR import zeroR


# Import the data
training = pd.read_csv("../training_init.csv")
X, y = numpyArrays(training)

# Correlation Matrix
print np.corrcoef(X[:, :12], rowvar=0)

# Set parameters
model = zeroR()
n = len(y)

# Perform 5-fold cross validation
scores = []
kf = KFold(n, n_folds=5, shuffle=True)

# Calculate root mean squared error for train/test for each fold
for train_idx, test_idx in kf:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    scores.append(rmse)

print "zeroR Model 5-Fold CV"
print scores
print np.mean(scores)

# Results:
# Scores: [1.2391598744748924, 1.2379526829331806, 1.2338724457782202, 1.2429822031605802, 1.2396314570248759]
# Average Score: 1.23871973267