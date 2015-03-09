__author__ = 'kaileyhoo'

import numpy as np
import pandas as pd
from data_processing import numpyArrays
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

def Round1(X, y):
    # Set parameters
    model = AdaBoostRegressor()
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
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        scores.append(rmse)

    print scores
    print np.mean(scores)
    return scores

if __name__ == "__main__":
    # Import the data
    training = pd.read_csv("../training_2.csv")

    training.drop('user_id', axis=1, inplace=True)

    X, y = numpyArrays(training)

    print "Boosting: AdaBoost Model 5-Fold CV"

    r1_scores = Round1(X, y)
    print r1_scores