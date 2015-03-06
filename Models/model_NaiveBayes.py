__author__ = 'griffin'

import numpy as np
import pandas as pd
from data_processing import numpyArrays
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB

# Import the data
training = pd.read_csv("../training_init.csv")
X, y = numpyArrays(training)

# Set parameters
model = GaussianNB()
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

print "Gaussian Naive Bayes Model 5-Fold CV"
print scores
print np.mean(scores)

# Results:
# Scores: [1.5784328682030409, 2.0411559941803801, 1.6033979142904309, 2.0879857887617321, 1.5821056576006167]
# Average Score: 1.77861564461