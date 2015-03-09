__author__ = 'griffin'

import pandas as pd
import numpy as np
import random
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import *
import operator
from data_cleaning import shuffle


def distance(user1_dict, user2_dict):
    common_businesses = list(set(user1_dict.keys()) & set(user2_dict.keys()))
    print len(common_businesses)
    ratings1 = np.asarray([user1_dict[bus_id] for bus_id in common_businesses])
    ratings2 = np.asarray([user2_dict[bus_id] for bus_id in common_businesses])
    return cosine(ratings1, ratings2)


if __name__ == "__main__":

    # Import the data
    data = pd.read_csv("../yelp_training.csv")
    data.drop_duplicates(['business_id', 'user_id'], inplace=True)
    ratings = data[['business_id', 'user_id', 'r_stars']]

    # Separate training, test
    n = len(ratings.index)
    n_train = int(0.8*n)

    # print "random splits"
    train_indices = random.sample(xrange(n), n_train)
    test_indices = list(set(xrange(n)) - set(train_indices))
    # print len(train_indices)
    # print len(test_indices)
    # print "separating"
    train = ratings.loc[train_indices, :]
    test = ratings.loc[test_indices, :]

    print "specific business"
    print train.loc[train['business_id'] == "MkDHjGBxz8r1mSjQf6kOUw", ]
    print test.loc[test['business_id'] == "MkDHjGBxz8r1mSjQf6kOUw", ]

    ratings_dict_train = {}
    ratings_dict_test = {}

    for idx, row in train.iterrows():
        if row['user_id'] in ratings_dict_train:
            ratings_dict_train[row['user_id']][row['business_id']] = row['r_stars']
        else:
            ratings_dict_train[row['user_id']] = {row['business_id']: row['r_stars']}

    for idx, row in test.iterrows():
        if row['user_id'] in ratings_dict_test:
            ratings_dict_test[row['user_id']][row['business_id']] = row['r_stars']
        else:
            ratings_dict_test[row['user_id']] = {row['business_id']: row['r_stars']}

    rec_scores = {}

    for test_user, test_ratings in ratings_dict_test.iteritems():
        print test_user, test_ratings
        distances = {}
        rec_scores[test_user] = {}
        for train_user, train_ratings in ratings_dict_train.iteritems():
            if train_user == test_user:
                continue
            elif len(list(set(test_ratings.keys()) & set(train_ratings.keys()))) == 0:
                continue
            else:
                distances[train_user] = distance(test_ratings, train_ratings)
        for bus_id in test_ratings.iterkeys():
            #print "Test User: " + str(test_user) + ", Business: " + str(bus_id)
            similar_users = [user_id for user_id, ratings in ratings_dict_train.iteritems() if bus_id in ratings]
            #print "    " + str(len(similar_users)) + " users also reviewed this business"
            similar_users_dist = {user_id: distances[user_id] for user_id in similar_users}
            top5 = sorted(similar_users_dist, key=similar_users_dist.get, reverse=True)[0:5]
            rec_score = np.sum([1.0/distances[user_id] * ratings_dict_train[user_id][bus_id] for user in top5])
            rec_scores[test_user][bus_id] = rec_score
        print "\n"

    # # Extract just the business_id, user_id, rating
    # ratings = training[['business_id', 'user_id', 'r_stars']]
    # #print ratings
    # ratings_wide = ratings.pivot(index='user_id', columns='business_id', values='r_stars')
    # #print ratings_wide.count(axis=1)
    # ratings_mat = ratings_wide.values
    # print ratings_mat.shape
    #
    # ratings_mat = ratings_mat[0:10, :]
    #
    # # Distance matrix between users
    # #dist = pdist(ratings_mat, 'correlation')
    #
    # dist = np.asarray(r.dist(ratings_mat))
    # print type(dist)
    # print dist

