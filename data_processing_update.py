__author__ = 'kaileyhoo'

import pandas as pd
import numpy as np
import re


# function to exclude all columns unnecessary/unusable or possible data leakage
# as well as update to make sure all columns are dummy variables
def updateColumns(df):

    for col in df.columns:
        # create dummy variables for all of the cities excluding last to avoid multicollinearity
        if col == u'b_city':
            df[col] = [re.sub(u' ', u'', city) for city in df[col]]
            cities_df = pd.get_dummies(df[col], prefix='b_city')
            cities_df.drop('b_city_Ahwatukee', axis=1, inplace=True)
            df = pd.concat([df, cities_df], axis=1)
            df.drop(col, axis=1, inplace=True)

        # change review stars to target column
        elif col == u'r_stars':
            df.rename(columns={col: 'target'}, inplace=True)

        # change date to epoch time
        elif col == u'r_date':
            df[col] = pd.to_datetime(df[col]).astype(np.int64) // 10**9

        # Change open/closed status from boolean to 1/0
        elif col == u'b_open':
            df[col] = df[col].astype(int)

        # list of columns we want to keep in our dataset as features
        elif col in ['b_latitude', 'b_longitude', 'b_review_count', 'b_stars_update'
                     'u_votes_cool', 'u_votes_funny', 'u_votes_useful', 'u_review_count', 'u_stars_update'] \
                or 'b_categories_' in col:
            continue

        # drop all other columns
        else:
            df.drop(col, axis=1, inplace=True)

    return df


# function to export user info for clustering
def clusterColumns(df):
    for col in df.columns:
        if col in ['user_id', 'u_votes_useful', 'u_review_count', 'u_stars_update']:
            continue
        # drop all other columns
        else:
            df.drop(col, axis=1, inplace=True)
    df.to_csv("userClustering.csv", index=False, encoding='utf-8')


# function to return two numpy arrays
# one of the arrays is the features and the other is the target variable
def numpyArrays(df):

    # create our target variable, and our feature variables
    target = 'target'
    features = [col for col in df.columns if col not in target]

    # set features to X and target to y numpy arrays
    X, y = np.array(df.ix[:, features]), np.array(df.ix[:, target])

    # shuffle the indices
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    return X, y


# Function to replace NA's in user data with values calculated
def updateMissing(df):
    missing_users = pd.read_csv('missingUserInfoTraining.csv')
    missing = missing_users.set_index('user_id').T.to_dict('dict')
    for i, row in df.iterrows():
        if np.isnan(row['u_average_stars']):
            if row['user_id'] in missing.keys():
                df.loc[i, 'u_average_stars'] = missing[row['user_id']]['u_average_stars']
                df.loc[i, 'u_review_count'] = missing[row['user_id']]['u_review_count']
                df.loc[i, 'u_votes_cool'] = missing[row['user_id']]['u_votes_cool']
                df.loc[i, 'u_votes_funny'] = missing[row['user_id']]['u_votes_funny']
                df.loc[i, 'u_votes_useful'] = missing[row['user_id']]['u_votes_useful']
            else:
                df.fillna(df.mean())
    return df


# function to create new columns that exclude the review star rating in the
# business and user average stars
def updateStars(df):
    df['b_stars_update'] = ((df['b_stars'] * df['b_review_count']) - df['r_stars']) / (df['b_review_count'] - 1)
    df['u_stars_update'] = ((df['u_average_stars'] * df['u_review_count']) - df['r_stars']) / (df['u_review_count'] - 1)
    return df


def commonDummies(train_df, test_df):
    """
    Force test set to have same dummy variables for city/business categories
    as the training set. Eliminate dummies in test that aren't in training,
    add column of zeros for dummies in training that aren't in test.
    :param train_df: Pandas dataframe
    :param test_df: Pandas dataframe
    :return: modified Pandas dataframes
    """

    # Find dummy variables for business city and categories
    dummies_train = [col for col in train_df.columns if re.search(u'b_categories_|b_city_', col) is not None]
    dummies_test = [col for col in test_df.columns if re.search(u'b_categories_|b_city', col) is not None]

    # Find dummy variables that are in one data set and not the other
    dummies_not_in_train = list(set(dummies_test) - set(dummies_train))
    dummies_not_in_test = list(set(dummies_train) - set(dummies_test))

    # Drop dummy variables from test that aren't in the training
    test_df.drop(dummies_not_in_train)

    # Add dummy variables to test that aren't already there, set them to 0
    for cat in dummies_not_in_test:
        test_df[cat] = 0

    # Sort columns so that they're in the same order for both data frames
    train_df.sort(axis=1, inplace=True)
    test_df.sort(axis=1, inplace=True)

    return train_df, test_df


if __name__ == '__main__':

    # Read CSV files to pandas dataframes
    training = pd.read_csv("yelp_training.csv")
    test = pd.read_csv("yelp_test.csv")

    # Impute missing values with new values calculated
    training = updateMissing(training)
    test = updateMissing(test)
    print "Updated missing data"

    # Redo business/user stars to avoid data leakage
    training = updateStars(training)
    test = updateStars(test)
    print "Updated stars to avoid data leakage"

    # Create a file for unique users for clustering
    clusterColumns(training)
    print "Exported file for user clustering"

    # Reduce variables, create dummies for categorical variables
    training = updateColumns(training)
    test = updateColumns(test)
    print "Created categorical dummy variables"

    # Force test data to have same dummy variable columns as training data
    training, test = commonDummies(training, test)
    print "Updated test data's dummy variables"

    # Save to new CSV
    print "Exporting to CSV"
    training.to_csv('training_2.csv', index=False, encoding='utf-8')
    test.to_csv('testing_2.csv', index=False, encoding='utf-8')
    print "Finished."