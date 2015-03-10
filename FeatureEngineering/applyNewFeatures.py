__author__ = 'griffin'

import pandas as pd
from data_processing_update import numpyArrays
from featureEngineering_CategoryReduction import reduceCategories
from featureEngineering_userClustering import userCluster
from featureEngineering_textFeatures import ngram_processing
from featureEngineering_CollaborativeFiltering import filtering

def applyFeatures(training_data, test_data, features_list, user_id_train, user_id_test):

    # Create numpy arrays
    X_train, y_train = numpyArrays(training_data)
    X_test, y_test = numpyArrays(test_data)

    if "Category Reduction" in features_list:
        # Reduce business categories using PCA
        category_col_indices = [idx for idx, col in enumerate(training_data.columns) if 'b_categories_' in col]
        X_train, X_test = reduceCategories(X_train, X_test, category_col_indices)
        print "Added Category Reduction"

    if "User Clustering" in features_list:
        # Add in column for user clusters
        X_train, X_test = userCluster(X_train, X_test, user_id_train)
        print "Added User Clustering"

    if "Text Features" in features_list:
        X_train, X_test = ngram_processing(X_train, X_test, user_id_train, user_id_test)
        print "Added Text Features"

    if "Collaborative Filtering" in features_list:
        X_train, X_test = filtering(X_train, X_test, user_id_train, user_id_test)
        print "Added Collaborative Filtering"

    return X_train, X_test
