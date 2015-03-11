__author__ = 'kaileyhoo'

import numpy as np
import pandas as pd
from data_processing_update import numpyArrays
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations
from FeatureEngineering.applyNewFeatures import applyFeatures
from data_cleaning import shuffle


# function to convert data frame to numpy arrays
# as well as drop user_id and business_id
def dfToArray(train_df, test_df):
    # # Strip out user_id with index
    # user_id = train_df['user_id']
    # user_id_test = test_df['user_id']
    #
    # # Strip out business_id with index
    # bus_id = train_df['business_id']
    # bus_id_test = test_df['business_id']

    # Drop user_id from dataframes
    train_update = train_df.drop(['user_id', 'business_id'], axis=1)
    test_update = test_df.drop(['user_id', 'business_id'], axis=1)

    # Create numpy arrays
    X_train_array, y_train_array = numpyArrays(train_update)
    X_test_array, y_test_array = numpyArrays(test_update)

    return X_train_array, y_train_array, X_test_array, y_test_array


# function to run cross validation on model
def round1(X, y):
    # Set parameters
    model = RandomForestRegressor()
    n = len(y)

    # Perform 5-fold cross validation
    scores = []
    kf = KFold(n, n_folds=5, shuffle=True)

    # Calculate root mean squared error for train/test for each fold
    for train_idx, test_idx in kf:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        prediction = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, prediction))
        scores.append(rmse)
    return scores


# function to apply all of the feature engineering into cross validation
def round2(X_df, featurelist):
    # Set parameters
    model = RandomForestRegressor(max_depth=50)
    y_df = X_df['target']
    n = len(y_df)

    # Perform 5-fold cross validation
    scores = []
    kf = KFold(n, n_folds=5, shuffle=True)

    # Calculate mean absolute deviation for train/test for each fold
    for train_idx, test_idx in kf:
        X_train, X_test = X_df.iloc[train_idx, :], X_df.iloc[test_idx, :]
        # y_train, y_test = y_df[train_idx], y_df[test_idx]

        X_train, X_test = applyFeatures(X_train, X_test, featurelist)
        Xtrain_array, ytrain_array, Xtest_array, ytest_array = dfToArray(X_train, X_test)
        model.fit(Xtrain_array, ytrain_array)
        prediction = model.predict(Xtest_array)
        rmse = np.sqrt(mean_squared_error(ytest_array, prediction))
        scores.append(rmse)

    return scores


if __name__ == "__main__":
    # Import the data
    training = pd.read_csv("../training_2.csv")
    training = shuffle(training)
    training = training[:int(.5*len(training))]
    test = pd.read_csv("../testing_2.csv")

    print "Random Forest Model 5-Fold CV"

    # X_train, y_train, X_test, y_test = dfToArray(training, test)
    #
    # r1_scores = round1(X_train, y_train)
    # print "No Features Added"
    # print "Scores:", r1_scores
    # print "Average Score:", np.mean(r1_scores)

    # Results
    # Scores: [0.861030105275716, 0.8763886629156924, 0.8612865947765096, 0.86778462553430558, 0.86877258317565187]
    # Average Score: 0.867052514336

    ### TEST MODEL WITH DIFFERENT FEATURES

    # # test on one feature
    # feature = ['User Clustering']
    # feature_scores = round2(training, feature)
    # print "Scores:", feature_scores
    # print "Average Score:", np.mean(feature_scores)

    # Test on individual features
    feature_eng = ['Category Reduction', 'User Clustering' , 'Text Features', 'Collaborative Filtering']
    for i in range(1, 5):
        combo = combinations(feature_eng, i)
        for com in combo:
            feature_list = list(com)
            print "Features:", feature_list
            feature_scores = round2(training, feature_list)
            print "Scores:", feature_scores
            print "Average Score:", np.mean(feature_scores)