__author__ = 'griffin'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import numpyArrays
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression


# Import the data
training = pd.read_csv("training_init.csv")
X, y = numpyArrays(training)

# Set parameters
model = LinearRegression()
n = len(y)

# Perform 5-fold cross validation
scores = []
kf = KFold(n, n_folds=5, shuffle=True)

# Calculate mean absolute deviation for train/test for each fold
for train_idx, test_idx in kf:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    mad = mean_absolute_error(y_test, prediction)
    scores.append(mad)

print scores
print np.mean(scores)