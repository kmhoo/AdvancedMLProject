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

        # exlude one of the categories to avoid multicollinearity
        elif col == u'b_categories_Accessories':
            df.drop(col, axis=1, inplace=True)

        # change date to epoch time
        elif col == u'r_date':
            df[col] = pd.to_datetime(df[col]).astype(np.int64) // 10**9

        # Change open/closed status from boolean to 1/0
        elif col == u'b_open':
            df[col] = df[col].astype(int)

        # list of columns we want to keep in our dataset as features
        elif col in ['b_latitude', 'b_longitude', 'b_review_count', 'b_stars',
                     'u_votes_cool', 'u_votes_funny', 'u_votes_useful', 'u_review_count', 'u_stars'] \
                or 'b_categories_' in col:
            continue

        # drop all other columns
        else:
            df.drop(col, axis=1, inplace=True)

    return df


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


# Function to impute missing values with the column mean
def roughImpute(df):
    df = df.fillna(df.mean())
    return df


if __name__=='__main__':
    training = pd.read_csv("yelp_training.csv")
    training = updateColumns(training)
    training = roughImpute(training)
    training.to_csv('training_init.csv', index=False, encoding='utf-8')
