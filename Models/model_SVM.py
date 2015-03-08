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
from sklearn.svm import SVR



# Import the data
training = pd.read_csv("../training_init.csv")
X, y = numpyArrays(training)

# Correlation Matrix
print np.corrcoef(X[:, :12], rowvar=0)

# Set parameters
model = SVR()
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
    print y_test
    print prediction
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    scores.append(rmse)

print "SVR Model 5-Fold CV"
print scores
print np.mean(scores)

# No results
# Takes too long to run...