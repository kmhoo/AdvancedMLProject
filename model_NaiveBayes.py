__author__ = 'griffin'

import numpy as np
import pandas as pd
from data_processing import numpyArrays
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB

# Import the data
training = pd.read_csv("training_init.csv")
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
    print y_test
    print prediction
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    scores.append(rmse)

print "Gaussian Naive Bayes Model 5-Fold CV"
print scores
print np.mean(scores)

# Gaussian Naive Bayes Model 5-Fold CV
# [2.0919294767440131, 2.0580977064684327, 2.0527344470003701, 1.5715536887101738, 1.5931116809819592]
# 1.87348539998