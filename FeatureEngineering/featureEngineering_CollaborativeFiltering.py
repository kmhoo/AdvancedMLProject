__author__ = 'griffin'

import pandas as pd
import random
from scipy.spatial.distance import *
import time


def distance(user1_dict, user2_dict):
    """
    Calculates distance between two users based on the businesses
    they have reviewed and the ratings
    :param user1_dict: dictionary of business_id: rating pairs
    :param user2_dict: dictionary of business_id: rating pairs
    :return: numeric distance
    """
    all_businesses = list(set(user1_dict.keys()) | set(user2_dict.keys()))
    ratings1 = np.asarray([1 if bus_id in user1_dict else 0 for bus_id in all_businesses])
    ratings2 = np.asarray([1 if bus_id in user2_dict else 0 for bus_id in all_businesses])
    return jaccard(ratings1, ratings2)


def createRatingsDictionary(ratings_df):
    """
    Create a dictionary of the form {user_id: {business_id: rating}} from a pandas
    data frame containing (at least) the columns user_id, business_id, rating (target)
    :param ratings_df: Pandas dataframe
    :return: dictionary of structure specified above
    """

    # Initialize ratings dictionary
    ratings_dict = {}

    # Add each row individually to the dictionary
    for idx, row in ratings_df.iterrows():
        # If the user is already in the dictionary, add the business and rating to their sub-dictionary
        if row['user_id'] in ratings_dict:
            ratings_dict[row['user_id']][row['business_id']] = row['target']
        # Otherwise, create their sub-dictionary and add the business/rating
        else:
            ratings_dict[row['user_id']] = {row['business_id']: row['target']}

    # Return dictionary
    return ratings_dict


def recommendationScores(ratings_dict_train, ratings_dict_test):
    """
    Calculates recommendation scores for each user/business combination in the
    test set using collaborative filtering.
    :param ratings_dict_test: Nested dictionary with structure {user_id: {business_id: rating}}
    :param ratings_dict_training: Nested dictionary with structure {user_id: {business_id: rating}}
    :return: nested dictionary, identical to ratings_dict_test except that it contains
    recommendation scores instead of ratings
    """

    # Initialize recommendation scores dictionary
    rec_scores = {}

    # For every user/business combo in the test set,
    # calculate recommendation score based on top 5 most similar users
    for test_user, test_ratings in ratings_dict_test.iteritems():
        #print test_user, test_ratings

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
            elif len(set(test_ratings.keys()) & set(train_ratings.keys())) == 0:
                continue
            # Calculate distance based on common business ratings
            else:
                distances[train_user] = distance(test_ratings, train_ratings)

        # For each business for the current user, calculate recommendation score
        for bus_id in test_ratings.iterkeys():

            # First narrow down similar users based on who has also reviewed the current question
            similar_users = [user_id for user_id, ratings in ratings_dict_train.iteritems() if bus_id in ratings]

            # Subset distance dictionary created in previous step to only hold these users
            similar_users_dist = {user_id: distances[user_id] for user_id in similar_users if user_id in distances}

            # Sort these users by distance to take the most similar 5
            # The "top 5" will have less than 5 if fewer than 5 other users have reviewed the business
            top5 = sorted(similar_users_dist, key=similar_users_dist.get, reverse=True)[0:5]

            # Calculate recommendation score as sum(similarity * rating) for top 5's ratings of this business
            rec_score_unscaled = np.sum([1.0/(distances[user_id]+0.1) * ratings_dict_train[user_id][bus_id] for user_id in top5])

            # Scale the score by the number of users who actually contributed to it
            rec_score = rec_score_unscaled/len(top5)

            # Add to recommendation scores dictionary
            rec_scores[test_user][bus_id] = rec_score

            # Print if there's a strange value
            if rec_score == 0 or rec_score == np.inf:
                print "Test User: " + str(test_user) + ", Business: " + str(bus_id)
                print "    " + str(len(similar_users_dist)) + " users also reviewed this business"
                print "    " + str(top5)
                print "    Recommendation Score: " + str(rec_score_unscaled) + "/" + str(len(top5)) + '=' + str(rec_score)
                print "\n"

    return rec_scores



def addRecommendationScores(train_df, test_df):
    """
    Function to execute collaborative filtering and append resulting
    recommendation scores to the training and test data frames
    :param train_df: Pandas data frame
    :param test_df: Pandas data frame
    :return: altered data frames
    """

    # Convert data frames to ratings dictionaries
    ratings_dict_train = createRatingsDictionary(train_df)
    ratings_dict_test = createRatingsDictionary(test_df)

    # Generate recommendation scores for training and test based on training
    rec_scores_test = recommendationScores(ratings_dict_train, ratings_dict_test)
    rec_scores_train = recommendationScores(ratings_dict_train, ratings_dict_train)

    # Add recommendation scores to training data frame
    train_df['rec_scores'] = np.nan
    for idx, row in train_df.iterrows():
        try:
            train_df.loc[idx, 'rec_scores'] = rec_scores_train[row['user_id']][row['business_id']]
        except KeyError, e:
            print e
            print "Could not find following record in dictionary:"
            print "User ID: " + str(row['user_id']) + ", Business ID: " + str(row['business_id'])
            test_df.loc[idx, 'rec_scores'] = np.nan

    # Add recommendation scores to test data frame
    test_df['rec_scores'] = np.nan
    for idx, row in test_df.iterrows():
        try:
            test_df.loc[idx, 'rec_scores'] = rec_scores_test[row['user_id']][row['business_id']]
        except KeyError, e:
            print e
            print "Could not find following record in dictionary:"
            print "User ID: " + str(row['user_id']) + ", Business ID: " + str(row['business_id'])
            test_df.loc[idx, 'rec_scores'] = np.nan

    # Impute remaining missing values with mean
    train_rec_mean = train_df['rec_scores'].mean()
    train_df['rec_scores'].fillna(train_rec_mean)
    test_df['rec_scores'].fillna(train_rec_mean)

    return train_df, test_df


if __name__ == "__main__":

    start = time.time()

    # Import the data
    data = pd.read_csv("../yelp_training.csv")
    data.drop_duplicates(['business_id', 'user_id'], inplace=True)
    #ratings = data[['business_id', 'user_id', 'r_stars']]

    # Separate training, test
    n = int(len(data.index)*0.01)
    n_train = int(0.8*n)
    train_indices = random.sample(xrange(n), n_train)
    test_indices = list(set(xrange(n)) - set(train_indices))
    train = data.loc[train_indices, ['user_id', 'business_id', 'r_stars', 'u_review_count']]
    test = data.loc[test_indices, ['user_id', 'business_id', 'r_stars', 'u_review_count']]

    # Spot tests
    # print "specific business"
    # print train.loc[train['business_id'] == "MkDHjGBxz8r1mSjQf6kOUw", ]
    # print test.loc[test['business_id'] == "MkDHjGBxz8r1mSjQf6kOUw", ]
    # print "]n"

    new_train, new_test = addRecommendationScores(train, test)

    end = time.time()
    #print "time elapsed: " + str(end-start)
