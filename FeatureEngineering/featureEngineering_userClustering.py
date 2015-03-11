__author__ = 'kaileyhoo'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from data_cleaning import shuffle


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
    print np.shape(train)
    # subset to only the user review information
    users_train = train.loc[:, ['user_id', 'u_votes_useful_update', 'u_review_count_update', 'u_stars_update']]
    # remove all duplicates and na values
    # unique_users = users_train.drop_duplicates(cols='user_id')
    # unique_users = unique_users.dropna()

    # only need features (drop user_id)
    features = users_train.loc[:, ['u_votes_useful_update', 'u_review_count_update', 'u_stars_update']]

    # scale the features
    feature_scaled = scale(np.array(features))

    # create clusters based on user_id level
    model = KMeans(n_clusters=5)
    model.fit(feature_scaled)
    users_train['cluster_labels'] = model.labels_

    unique_users = users_train.loc[:, ['cluster_labels']]

    # add clusters into original data
    train_update_df = pd.concat([train, unique_users], axis=1)

    clusters_df = pd.get_dummies(train_update_df['cluster_labels'], prefix='cluster')
    clusters_df.drop('cluster_0', axis=1, inplace=True)
    train_update_df = pd.concat([train_update_df, clusters_df], axis=1)
    train_update_df.drop('cluster_labels', axis=1, inplace=True)
    # print np.shape(train_update_df)

    # print np.shape(test)
    # subset to only review information
    users_test = test.loc[:, ['u_votes_useful_update', 'u_review_count_update', 'u_stars_update']]

    # use fitted model to add clusters to test data
    test_feature_scale = scale(np.array(users_test))

    test_update_df = test
    test_update_df['cluster_labels'] = model.predict(test_feature_scale)

    # create dummy variables for clusters
    test_clusters_df = pd.get_dummies(test_update_df['cluster_labels'], prefix='cluster')
    test_clusters_df.drop('cluster_0', axis=1, inplace=True)
    test_update_df = pd.concat([test_update_df, test_clusters_df], axis=1)
    test_update_df.drop('cluster_labels', axis=1, inplace=True)

    return train_update_df, test_update_df


if __name__ == "__main__":
    training = pd.read_csv("../training_2.csv")
    training = shuffle(training)
    training = training[:int(.5*len(training))]
    test = pd.read_csv("../testing_2.csv")
    print np.shape(training), np.shape(test)
    train_df, test_df = userCluster(training, test)
    print np.shape(train_df), np.shape(test_df)