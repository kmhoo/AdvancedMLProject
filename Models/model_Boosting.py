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


def Round2(X, y):
    # Set parameters
    min_score = {}
    for loss in ['linear', 'square', 'exponential']:
        model = AdaBoostRegressor(loss=loss)
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
            # score = model.score(X_test, y_test)
            scores.append(rmse)
        if len(min_score) == 0:
            min_score['loss'] = loss
            min_score['scores'] = scores
        else:
            if np.mean(scores) < np.mean(min_score['scores']):
                min_score['loss'] = loss
                min_score['scores'] = scores

        print "Loss:", loss
        print scores
        print np.mean(scores)
    return min_score


if __name__ == "__main__":
    # Import the data
    training = pd.read_csv("../training_init.csv")
    X, y = numpyArrays(training)

    print "Boosting: AdaBoost Model 5-Fold CV"

    r1_scores = Round1(X, y)
    print r1_scores
    # No hyperparameters chosen
    # Scores: [1.0447293207328756, 1.0587182333258511, 1.031543751312116, 1.0481367745409615, 1.0204516830079458]
    # Average Score: 1.04071595258

    r2_scores = Round2(X, y)
    print r2_scores
    # Tuning hyperparameters: loss
    # Best Fit: linear
    # Scores: [1.0373288075252349, 1.0167692841530487, 1.0490664742021374, 1.0699056621006253, 1.0399466268061142]
    # Average Score: 1.04260337096
