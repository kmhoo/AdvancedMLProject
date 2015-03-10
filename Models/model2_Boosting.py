__author__ = 'kaileyhoo'

import numpy as np
import pandas as pd
from data_processing import numpyArrays
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from itertools import combinations
from FeatureEngineering.applyNewFeatures import applyFeatures


def round1(X, y):
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

    return scores

if __name__ == "__main__":
    # Import the data
    training = pd.read_csv("../training_2.csv")
    test = pd.read_csv("../testing_2.csv")

    # Strip out user_id with index
    user_id = training['user_id']
    user_id_test = test['user_id']

    # Strip out business_id with index
    bus_id = training['business_id']
    bus_id_test = test['business_id']

    # Drop user_id from dataframes
    training.drop('user_id', axis=1, inplace=True)
    training.drop('business_id', axis=1, inplace=True)
    test.drop('user_id', axis=1, inplace=True)
    test.drop('business_id', axis=1, inplace=True)

    # Create numpy arrays
    X_train, y_train = numpyArrays(training)
    X_test, y_test = numpyArrays(test)

    print "Boosting: AdaBoost Model 5-Fold CV"

    r1_scores = round1(X_train, y_train)
    print r1_scores
    print np.mean(r1_scores)

    ### TEST MODEL WITH DIFFERENT FEATURES

    # Test on individual features
    feature_eng = ['Category Reduction', 'User Clustering', 'Text Features', 'Collaborative Filtering']
    for i in range(1, 5):
        combo = combinations(feature_eng, i)
        for com in combo:
            feature_list = list(com)
            print feature_list
            X_train, X_test = applyFeatures(training, test, feature_list, user_id, user_id_test)
            print "Added features listed above"

            feature_scores = round1(X_train, y_train)
            print "Scores:", feature_scores
            print "Average Score:", np.mean(feature_scores)