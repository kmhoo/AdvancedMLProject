__author__ = 'kaileyhoo'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


def userCluster(train, test):
    """
    Clusters uses based on useful votes (do not need funny and cool since all three are extremely
    correlated), review count, and users' average stars. Adds the user clusters to the array and
    drops the user id
    :param train: numpy array of training data without user_ids
    :param test: numpy array of test data
    :param users: df of user_ids
    :return: numpy arrays without user_id
    """

    # subset to only the user review information
    users_train = train.loc[:, ['user_id', 'u_votes_useful_update', 'u_review_count_update', 'u_stars_update']]
    # remove all duplicates and na values
    unique_users = users_train.drop_duplicates(cols='user_id')
    unique_users = unique_users.dropna()

    # only need features (drop user_id)
    features = unique_users.loc[:, ['u_votes_useful_update', 'u_review_count_update', 'u_stars_update']]

    # scale the features
    feature_scaled = scale(np.array(features))

    # create clusters based on user_id level
    model = KMeans(n_clusters=5)
    model.fit(feature_scaled)
    unique_users['cluster_labels'] = model.labels_

    unique_users = unique_users.loc[:, ['user_id', 'cluster_labels']]

    # add clusters into original data
    train_update_df = pd.merge(train, unique_users, on='user_id', how='left')

    # subset to only review information
    users_test = test.loc[:, ['u_votes_useful_update', 'u_review_count_update', 'u_stars_update']]

    # use fitted model to add clusters to test data
    test_feature_scale = scale(np.array(users_test))

    test_update_df = test
    test_update_df['cluster_labels'] = model.predict(test_feature_scale)

    return train_update_df, test_update_df