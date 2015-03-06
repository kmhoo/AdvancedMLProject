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
# Scores: [1.0805854136050308, 1.0808996104801534, 1.0822870751664566, 1.0824667286776666, 1.083302262583459]
# Average Score: 1.0819082181


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
# Scores: [1.0806288393228796, 1.0808595682114084, 1.0821777076167756, 1.0823900435636453, 1.0831126039732337]
# Average Score: 1.08183375254