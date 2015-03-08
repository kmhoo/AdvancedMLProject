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
from sklearn.ensemble import RandomForestRegressor

def round1(X, y):
    # Set parameters
    model = RandomForestRegressor()
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
    print scores
    print np.mean(scores)
    return scores


def round2(X, y):
    # Set parameters
    min_score = {}
    for tree in [50, 100, 200, 500]:
        for feature in ['auto', 'log2']:
            model = RandomForestRegressor(n_estimators=tree, max_features=feature)
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
            if len(min_score) == 0:
                min_score['estimator'] = tree
                min_score['max_feature'] = feature
                min_score['scores'] = scores
            else:
                if np.mean(scores) < np.mean(min_score['scores']):
                    min_score['estimator'] = tree
                    min_score['max_feature'] = feature
                    min_score['scores'] = scores

            print "Estimator:", tree
            print "Max Features:", feature
            print scores
            print np.mean(scores)
    return min_score


if __name__ == "__main__":
    # Import the data
    training = pd.read_csv("../training_init.csv")
    X, y = numpyArrays(training)

    # Correlation Matrix
    print np.corrcoef(X[:, :12], rowvar=0)

    print "Random Forest Model 5-Fold CV"

    r1_scores = round1(X, y)
    print r1_scores
    # No hyperparameters chosen
    # Scores: [1.034907964781089, 1.0377352352079787, 1.0297716060662643, 1.0285783021199912, 1.0380972759182057]
    # Average Score: 1.03381807682

    r2_scores = round2(X, y)
    print r2_scores
    # Tuning hyperparameters: number of trees, features per split
    # Best Fit: 100 trees, auto features
    # Scores: [1.0818922781167881, 1.0870302997861752, 1.0859522148892891, 1.090698304014788, 1.0877505371306078]
    # Average Score: 1.08666472679