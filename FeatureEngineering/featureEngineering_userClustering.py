__author__ = 'kaileyhoo'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


def userCluster(train, test, users):
    """
    Clusters uses based on useful votes (do not need funny and cool since all three are extremely
    correlated), review count, and users' average stars. Adds the user clusters to the array and
    drops the user id
    :param train: numpy array of training data without user_ids
    :param test: numpy array of test data
    :param users: df of user_ids
    :return: numpy arrays without user_id
    """
    # convert data back into dataframes
    users_train = pd.DataFrame(train)
    users_test = pd.DataFrame(test)

    # add users back into training set
    users_train['user_id'] = users

    # subset to only the user review information
    users_train = users_train.loc[:, ['user_id', 'u_votes_useful_update', 'u_review_count_update', 'u_stars_update']]
    users_test = users_test.loc[:, ['u_votes_useful_update', 'u_review_count_update', 'u_stars_update']]

    # remove all duplicates and na values
    unique_users = users_train.drop_duplicates()
    unique_users = unique_users.dropna()

    # only need features (drop user_id)
    features = unique_users.loc[:, ['u_votes_useful', 'u_review_count', 'u_stars_update']]

    # scale the features
    feature_scaled = scale(np.array(features))

    # create clusters based on user_id level
    model = KMeans(n_clusters=5)
    model.fit(feature_scaled)
    unique_users['cluster_labels'] = model.labels_

    # add clusters into original data
    users_train_update = pd.merge(users_train, unique_users, on='user_id', how='left')
    users_train_update.drop('user_id', axis=1, inplace=True)
    np_train = np.array(users_train_update)

    # use fitted model to add clusters to test data
    test_feature_scale = scale(np.array(users_test))
    users_test['cluster_labels'] = model.predict(test_feature_scale)
    np_test = np.array(users_test)

    return np_train, np_test