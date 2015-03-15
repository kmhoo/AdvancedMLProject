__author__ = 'griffin'

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor


def reduceCategories(train_df, test_df, category_col_names):
    """
    Uses PCA to reduce the number of features for business category to 100.
    Fits principal components on training data and transforms both the
    training and test data to replace category features with these components.
    :param train_df: Pandas dataframe for training data
    :param test_df: Pandas dataframe for test data
    :param col_indices: which columns to use in PCA
    :return: both arrays with replaced columns
    """

    num_components = 100

    # Columns to keep/not alter during PCA process
    column_names = train_df.columns
    keep_columns = [name for name in train_df.columns if name not in category_col_names]

    # Subset train/test data to only use specified category columns
    categories_train = np.asarray(train_df.loc[:, category_col_names])
    categories_test = np.asarray(test_df.loc[:, category_col_names])

    # Fit PCA on the training categories
    pca = PCA(n_components=num_components)
    pca.fit(categories_train)

    # Transform training and test categories into components
    components_train = pca.transform(categories_train)
    components_test = pca.transform(categories_test)

    # Change components to pandas data frames with column names
    component_names = ['b_categories_pc'+str(i) for i in xrange(num_components)]
    components_train = pd.DataFrame(components_train, columns=component_names).reset_index()
    components_test = pd.DataFrame(components_test, columns=component_names).reset_index()

    # Remove original category columns
    train_df_new = train_df.loc[:, keep_columns].reset_index()
    test_df_new = test_df.loc[:, keep_columns].reset_index()

    # Append component columns
    train_df_new2 = pd.concat([train_df_new, components_train], axis=1)
    test_df_new2 = pd.concat([test_df_new, components_test], axis=1)

    # Fill nas with the mean
    train_df_new2.fillna(train_df_new2.mean(), inplace=True)
    test_df_new2.fillna(test_df_new2.mean(), inplace=True)

    return train_df_new2, test_df_new2


if __name__ == "__main__":

    # Import the data
    data = pd.read_csv("../training_init.csv")
    category_col = [col for col in data.columns if 'b_categories_' in col]

    ## Test the reduceCategories function
    n = len(data.index)
    n_train = int(0.95*n)
    train_indices = random.sample(xrange(n), n_train)
    test_indices = list(set(xrange(n)) - set(train_indices))
    train_df = data.loc[train_indices, :]
    test_df = data.loc[test_indices, :]

    train_df_new, test_df_new = reduceCategories(train_df, test_df, category_col)

    # Extract business category columns
    categories = data[category_col]
    categories = np.asarray(categories)

    # ## Analyze variance inflation factor (VIF) for every variable
    # VIFs = [variance_inflation_factor(categories, i) for i in np.arange(categories.shape[0])]
    #
    # # Plot histogram of VIFs
    # plt.hist(VIFs, color=(150, 0, 205))
    # plt.title("Histogram of Variance Inflation Factors Among Category Features")
    # plt.xlabel("VIF")
    # plt.ylabel("Frequency")
    # plt.savefig("vif_hist.png")
    # plt.close()

    ## Create a scree plot

    # Fit the PCA on full training
    pca = PCA()
    pca.fit(categories)

    # Calculate cumulative sum of explained variance ratio and determine how many components
    # are necessary to explain 90% of the variance
    explained_var_cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = next(idx for idx, value in enumerate(explained_var_cumsum) if value > 0.9)

    # Scree plot
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(range(len(explained_var_cumsum)), explained_var_cumsum)
    plt.hlines(y=0.9, xmin=0, xmax=n_components, colors='red', linestyles='dashed')
    plt.vlines(x=n_components, ymin=0, ymax=0.9, colors='red', linestyles='dashed')
    plt.ylim(0, 1)
    plt.xlim(0, 508)
    plt.subplots_adjust(top=0.85)
    plt.title('Proportion of Variance Explained\nby Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Proportion of Variance Explained')
    plt.savefig("pca_scree.png")
    plt.close()
