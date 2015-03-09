__author__ = 'jrbaker'

import numpy as np
import pandas as pd
from data_processing import numpyArrays
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
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

    # # Correlation Matrix
    # print np.corrcoef(X[:, :12], rowvar=0)

    print "Random Forest Model 5-Fold CV"

    r1_scores = round1(X, y)
    print r1_scores
    # No hyperparameters chosen
    # Scores: [1.0157712279500486, 1.0301022982543202, 1.0280147414655603, 1.0314889024081446, 1.0271965671277661]
    # Average Score: 1.02651474744

    r2_scores = round2(X, y)
    print r2_scores
    # Tuning hyperparameters: number of trees, features per split
    # Best Fit: 100 trees, auto features
    # Scores: [0.98647614117542015, 0.98888481600757883, 0.98525012244656129, 0.98294018181240805, 0.98826424741039065]
    # Average Score: 0.98636310177