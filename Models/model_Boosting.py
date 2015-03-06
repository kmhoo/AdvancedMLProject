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
    # Scores: [1.1081784623612589, 1.1110850689273575, 1.1133579472724211, 1.1096344452553069, 1.0917598587826918]
    # Average Score: 1.10680315652

    r2_scores = Round2(X, y)
    print r2_scores
    # Tuning hyperparameters: loss
    # Best Fit: linear
    # Scores: [1.081736651278649, 1.0880794989600404, 1.0936843071987479, 1.1140979195964209, 1.0905866047574622]
    # Average Score: 1.09363699636
