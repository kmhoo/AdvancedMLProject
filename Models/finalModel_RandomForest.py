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
import pickle


# function to convert data frame to numpy arrays
# as well as drop user_id and business_id
def dfToArray(train_df, test_df):
    # Drop user_id from dataframes
    train_update = train_df.drop(['user_id', 'business_id'], axis=1)
    test_update = test_df.drop(['user_id', 'business_id'], axis=1)

    # Create numpy arrays
    X_train_array, y_train_array = numpyArrays(train_update)
    X_test_array, y_test_array = numpyArrays(test_update)

    return X_train_array, y_train_array, X_test_array, y_test_array


# function to apply all of the feature engineering into cross validation
# while tuning hyper-parameters
def hyper_parameter_tuning(X_df, featurelist):
    # Set parameters
    y_df = X_df['target']
    n = len(y_df)

    hyperparameters = []

    # go through different iterations of parameters
    for tree in [10, 50, 100, 200, 500, 1000]:
        for feature in ['auto', 'sqrt', 'log2']:
            parameter = {}
            parameter['tree'] = tree
            parameter['max_features'] = feature

            # create model
            model = RandomForestRegressor(n_estimators=tree, max_features=feature, max_depth=50, n_jobs=-1)

            print "Num of Trees:", tree
            print "Max Features:", feature

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
                print "Finish fold"

            print "Average Score:", np.mean(scores)
            # add to dictionary
            parameter['scores'] = scores
            parameter['mean_score'] = np.mean(scores)

            # add to list of different hyper-parameters
            hyperparameters.append(parameter)

    return hyperparameters


def final(training, test, features, parameters):
    # Set parameters
    model = RandomForestRegressor(n_estimators=parameters['tree'],
                                  max_features=parameters['max_features'],
                                  max_depth=50,
                                  n_jobs=-1)

    # apply features
    X_train, X_test = applyFeatures(training, test, features)
    Xtrain_array, ytrain_array, Xtest_array, ytest_array = dfToArray(X_train, X_test)
    model.fit(Xtrain_array, ytrain_array)

    # pickle final model
    pklfile = open("FinalModel.pkl", "w")
    pickle.dumps(model, pklfile)

    # predict the test data
    prediction = model.predict(Xtest_array)

    # calculate rmse
    rmse = np.sqrt(mean_squared_error(ytest_array, prediction))

    return rmse



if __name__ == "__main__":
    # Import the data
    training = pd.read_csv("../training_2.csv")
    training = shuffle(training)
    training = training[:int(0.7*len(training))]
    test = pd.read_csv("../testing_2.csv")

    print "Random Forest Model 5-Fold CV:Final"

    # Tuning hyper-parameters to find best with features below
    feature_eng = ['Category Reduction', 'User Clustering']
    print "Features:", feature_eng
    hp_list = hyper_parameter_tuning(training, feature_eng)

    # find best combination of hyperparameters
    sort_list = sorted(hp_list, key=lambda k: k['mean_score'])
    best_model = sort_list[0]
    print best_model

    # do the final model and get results
    final_rmse = final(training, test, feature_eng, best_model)
    print "Final RMSE of True Test:", final_rmse
