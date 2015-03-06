__author__ = 'griffin'

import pandas as pd
from data_processing import numpyArrays
from featureEngineering_CategoryReduction import reduceCategories

# Import the data
training = pd.read_csv("../training_init.csv")
test = pd.read_csv("../testing_init.csv")

X_train, y_train = numpyArrays(training)
X_test, y_test = numpyArrays(test)

# Reduce business categories using PCA
category_col_indices = [idx for idx, col in enumerate(training.columns) if 'b_categories_' in col]
X_train, X_test = reduceCategories(X_train, X_test, category_col_indices)

