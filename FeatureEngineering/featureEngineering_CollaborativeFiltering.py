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
    #print len(common_businesses)
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

    # For every user/business combo in the test set,
    # calculate recommendation score based on top 5 most similar users
    for test_user, test_ratings in ratings_dict_test.iteritems():
        print test_user, test_ratings

        # Initialize dictionary to hold distances between current test user and all training users
        distances = {}
        # Initialize key-value pair in recommendation scores dictionary to hold the
        # recommendation scores for this user for all businesses
        rec_scores[test_user] = {}

        # For each training user, calculate the distance between them and current test user
        for train_user, train_ratings in ratings_dict_train.iteritems():
            # Do not consider the same user
            if train_user == test_user:
                continue
            # Do not consider users who haven't reviewed any of the same businesses
            elif len(list(set(test_ratings.keys()) & set(train_ratings.keys()))) == 0:
                continue
            # Calculate distance based on common business ratings
            else:
                distances[train_user] = distance(test_ratings, train_ratings)

        # For each business for the current user, calculate recommendation score
        for bus_id in test_ratings.iterkeys():
            print "Test User: " + str(test_user) + ", Business: " + str(bus_id)
            # First narrow down similar users based on who has also reviewed the current question
            similar_users = [user_id for user_id, ratings in ratings_dict_train.iteritems() if bus_id in ratings]
            print "    " + str(len(similar_users)) + " users also reviewed this business"
            # Subset distance dictionary created in previous step to only hold these users
            similar_users_dist = {user_id: distances[user_id] for user_id in similar_users}
            # Sort these users by distance to take the most similar 5
            top5 = sorted(similar_users_dist, key=similar_users_dist.get, reverse=True)[0:5]
            # Calculate recommendation score as sum(similarity * rating) for top 5's ratings of this business
            rec_score = np.sum([1.0/distances[user_id] * ratings_dict_train[user_id][bus_id] for user_id in top5])
            # Add to recommendation scores dictionary
            rec_scores[test_user][bus_id] = rec_score
        print "\n"


    # For every user/business combo in the test set,
    # calculate recommendation score based on top 5 most similar users
    for train_user1, train_ratings1 in ratings_dict_train.iteritems():
        print train_user1, train_ratings1

        # Initialize dictionary to hold distances between current test user and all training users
        distances = {}
        # Initialize key-value pair in recommendation scores dictionary to hold the
        # recommendation scores for this user for all businesses
        rec_scores[train_user1] = {}

        # For each training user, calculate the distance between them and current test user
        for train_user, train_ratings in ratings_dict_train.iteritems():
            # Do not consider the same user
            if train_user == train_user1:
                continue
            # Do not consider users who haven't reviewed any of the same businesses
            elif len(list(set(train_ratings1.keys()) & set(train_ratings.keys()))) == 0:
                continue
            # Calculate distance based on common business ratings
            else:
                distances[train_user] = distance(train_ratings1, train_ratings)

        # For each business for the current user, calculate recommendation score
        for bus_id in train_ratings1.iterkeys():
            print "Test User: " + str(train_user1) + ", Business: " + str(bus_id)
            # First narrow down similar users based on who has also reviewed the current question
            similar_users = [user_id for user_id, ratings in ratings_dict_train.iteritems()
                             if bus_id in ratings and user_id != train_user1]
            print "    " + str(len(similar_users)) + " users also reviewed this business"
            # Subset distance dictionary created in previous step to only hold these users
            similar_users_dist = {user_id: distances[user_id] for user_id in similar_users}
            # Sort these users by distance to take the most similar 5
            top5 = sorted(similar_users_dist, key=similar_users_dist.get, reverse=True)[0:5]
            # Calculate recommendation score as sum(similarity * rating) for top 5's ratings of this business
            rec_score = np.sum([1.0/distances[user_id] * ratings_dict_train[user_id][bus_id] for user_id in top5])
            # Add to recommendation scores dictionary
            rec_scores[train_user1][bus_id] = rec_score
        print "\n"

