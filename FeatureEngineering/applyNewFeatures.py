__author__ = 'griffin'

import pandas as pd
from data_processing_update import numpyArrays
from featureEngineering_CategoryReduction import reduceCategories
from featureEngineering_userClustering import userCluster
from featureEngineering_textFeatures import ngram_processing
from featureEngineering_CollaborativeFiltering import filtering

def applyFeatures(training_data, test_data, features_list):

    if "Category Reduction" in features_list:
        # Reduce business categories using PCA
        category_col_indices = [col for col in training_data.columns if 'b_categories_' in col]
        training_data, X_test = reduceCategories(training_data, test_data, category_col_indices)
        print "Added Category Reduction"

    if "User Clustering" in features_list:
        # Add in column for user clusters
        training_data, test_data = userCluster(training_data, test_data)
        print "Added User Clustering"

    if "Text Features" in features_list:
        training_data, test_data = ngram_processing(training_data, test_data)
        print "Added Text Features"

    if "Collaborative Filtering" in features_list:
        training_data, test_data = filtering(training_data, test_data)
        print "Added Collaborative Filtering"

    return training_data, test_data
