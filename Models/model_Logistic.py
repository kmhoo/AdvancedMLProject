__author__ = 'kaileyhoo'

import numpy as np
import pandas as pd
from data_processing import numpyArrays
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


def round1(X, y):
    # Set parameters
    model = LogisticRegression()
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

    print scores
    print np.mean(scores)
    return scores


def round2(X, y):
    # Set parameters
    min_score = []
    for l in ["l1", "l2"]:
        for c in np.arange(0.1, 5.0, step=0.5):
            model = LogisticRegression(penalty=l, C=c)
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
            if len(min_score) == 0:
                min_score['penalty'] = l
                min_score['c_value'] = c
                min_score['scores'] = scores
            else:
                if np.mean(scores) < np.mean(min_score['scores']):
                    min_score['penalty'] = l
                    min_score['c_value'] = c
                    min_score['scores'] = scores

            print "Penalty:", l
            print "C:", c
            print scores
            print np.mean(scores)
    return min_score


if __name__ == "__main__":
    # Import the data
    training = pd.read_csv("../training_init.csv")
    X, y = numpyArrays(training)

    r1_scores = round1(X, y)
    print r1_scores
    # No hyperparameters chosen
    # [1.2420961206240748, 1.2426431597862457, 1.2283077690601429, 1.24061791003319, 1.2386332446702006]
    # 1.23845964083


    r2_scores = round2(X, y)
    print r2_scores
    # Tuning hyperparameters: penalty, C
    # Takes too long to run...
