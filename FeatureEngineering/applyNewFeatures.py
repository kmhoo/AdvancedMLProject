__author__ = 'griffin'

import pandas as pd
from data_processing_update import numpyArrays
from featureEngineering_CategoryReduction import reduceCategories
from featureEngineering_userClustering import userCluster

# Import the data
training = pd.read_csv("../training_2.csv")
test = pd.read_csv("../testing_2.csv")

# Create dictionary of user_id with index
user_id = training['user_id']

# Drop user_id from dataframes
training.drop('user_id', axis=1, inplace=True)
test.drop('user_id', axis=1, inplace=True)

# Create numpy arrays
X_train, y_train = numpyArrays(training)
X_test, y_test = numpyArrays(test)

# Reduce business categories using PCA
category_col_indices = [idx for idx, col in enumerate(training.columns) if 'b_categories_' in col]
X_train, X_test = reduceCategories(X_train, X_test, category_col_indices)

# Add in column for user clusters
X_train, X_test = userCluster(X_train, X_test, user_id)