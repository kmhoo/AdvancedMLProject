__author__ = 'griffin'

import numpy as np
import pandas as pd
from data_processing import numpyArrays
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor


# Import the data
training = pd.read_csv("../training_init.csv")
X, y = numpyArrays(training)


# Set parameters
model = LinearRegression()
n = len(y)
scores = []
kf = KFold(n, n_folds=5, shuffle=True)

# Perform 5-fold cross validation
# Calculate root mean squared error for train/test for each fold
for train_idx, test_idx in kf:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    scores.append(rmse)

print "Linear Model 5-Fold CV"
print scores
print np.mean(scores)

# Results:
# [1.0842133286956541, 1.0797514724220016, 1.0776218329524856, 1.086510788978603, 1.0851177348939898]
# 1.08264303159


## Bagged OLS Regressions

# Set parameters
model = BaggingRegressor(LinearRegression())
scores = []

# Perform 5-fold cross validation
# Calculate root mean squared error for train/test for each fold
for train_idx, test_idx in kf:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    scores.append(rmse)

print "Bagged Linear Model 5-Fold CV"
print scores
print np.mean(scores)

# Results
# Scores: [1.0842974720764995, 1.0794934687518074, 1.0774446462294078, 1.0866156210108342, 1.0852817744089476]
# Average Score: 1.0826265965