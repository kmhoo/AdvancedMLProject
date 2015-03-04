__author__ = 'griffin'

import numpy as np
import pandas as pd
from data_processing import numpyArrays
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor


# Import the data
training = pd.read_csv("training_init.csv")
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
    print y_test
    print prediction
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    scores.append(rmse)

print "Bagged Linear Model 5-Fold CV"
print scores
print np.mean(scores)