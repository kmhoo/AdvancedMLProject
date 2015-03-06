__author__ = 'griffin'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_processing import numpyArrays

def reduceCategories(train_arr, test_arr, col_indices):
    """
    Uses PCA to reduce the number of features for business category to 100.
    Fits principal components on training data and transforms both the
    training and test data to replace category features with these components.
    :param train_arr: Numpy array for training data
    :param test_arr: Numpy array for test data
    :param col_indices: which columns to use in PCA
    :return: both arrays with replaced columns
    """

    # Subset train/test data to only use specified category columns
    categories_train = train_arr[:, col_indices]
    categories_test = test_arr[:, col_indices]

    # Fit PCA on the training categories
    pca = PCA(n_components=100)
    pca.fit(categories_train)

    # Transform training and test categories into components
    components_train = pca.transform(categories_train)
    components_test = pca.transform(categories_test)
    print components_train

    # Remove original category columns
    train_arr_new = np.delete(train_arr, col_indices, axis=1)
    test_arr_new = np.delete(test_arr, col_indices, axis=1)

    # Append component columns
    train_arr_new = np.hstack((train_arr_new, components_train))
    test_arr_new = np.hstack((test_arr_new, components_test))

    return train_arr_new, test_arr_new


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
    plt.savefig("pca_scree.png")
    plt.close()
