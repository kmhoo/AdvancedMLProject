__author__ = 'kaileyhoo'

import numpy as np
import pandas as pd
from data_processing import numpyArrays
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

def Round1(X, y):
    # Set parameters
    model = KNeighborsRegressor()
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
    for neigh in [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:

        model = KNeighborsRegressor(n_neighbors=neigh)
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
            min_score['neighbor'] = neigh
            min_score['scores'] = scores
        else:
            if np.mean(scores) < np.mean(min_score['scores']):
                min_score['neighbor'] = neigh
                min_score['scores'] = scores
        print "Neighbors:", neigh
        print scores
        print np.mean(scores)
    return min_score


if __name__ == "__main__":
    # Import the data
    training = pd.read_csv("../training_init.csv")
    X, y = numpyArrays(training)

    print "KNeighbors Model 5-Fold CV"

    r1_scores = Round1(X, y)
    print r1_scores
    # No hyperparameters chosen
    # Scores: [1.3027852556422643, 1.3161003600307102, 1.3113649872070021, 1.3152631241575412, 1.3061084413509341]
    # Average Score: 1.31032443368

    r2_scores = Round2(X, y)
    print r2_scores
    # Tuning hyperparameters: n_neighbors
    # Best Fit: Neighbors: 2000
    # Scores: [1.2171823728773263, 1.2106270371000836, 1.2199629716380753, 1.2181826660717305, 1.2157755467101039]
    # Average Score: 1.21634611888
