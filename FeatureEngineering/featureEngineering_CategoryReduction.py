__author__ = 'griffin'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_processing import numpyArrays

def reduceCategories(train_df, test_df):
    """
    Uses PCA to reduce the number of features for business category from
    507 to 100. Fits principal components on training data and transforms
    both the training and test data to replace category features with these components.
    :param train_df: Pandas dataframe for training data
    :param test_df: Pandas dataframe for test data
    :return: both dataframes with replaced columns
    """

    # Find names of business category columns
    category_col = [col for col in train_df.columns if 'b_categories_' in col]

    # Extract business category columns, convert to numpy arrays
    categories_train = np.asarray(train_df[category_col])
    categories_test = np.asarray(test_df[category_col])

    # Fit PCA on the training categories
    pca = PCA()
    pca.fit(categories_train)

    # Transform training and test categories into components
    components_train = pca.transform(categories_train)
    components_test = pca.transform(categories_test)
    print components_train

    # Append new


if __name__ == "__main__":

    # Import the data
    training = pd.read_csv("../training_init.csv")

    # Extract business category columns
    category_col = [col for col in training.columns if 'b_categories_' in col]
    categories = training[category_col]
    categories = np.asarray(categories)

    # Fit the PCA on full training
    pca = PCA()
    pca.fit(categories)

    # Calculate cumulative sum of explained variance ratio and determine how many components
    # are necessary to explain 90% of the variance
    explained_var_cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = next(idx for idx, value in enumerate(explained_var_cumsum) if value > 0.9)

    # Scree plot
    plt.plot(range(len(explained_var_cumsum)), explained_var_cumsum)
    plt.hlines(y=0.9, xmin=0, xmax=n_components, colors='red', linestyles='dashed')
    plt.vlines(x=n_components, ymin=0, ymax=0.9, colors='red', linestyles='dashed')
    plt.ylim(0, 1)
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Proportion of Variance Explained')
    plt.savefig("pca.png")
    plt.close()
