__author__ = 'kaileyhoo'

import pandas as pd
import json
import re
import random
import numpy as np

# function to open the JSON file and return it as a dataframe
def OpenFile(path):
    data = open(path, 'r')
    json_files = []
    for jsonfile in data:
        json_files.append(json.loads(jsonfile))
    filename = pd.DataFrame(json_files)
    return filename

# rename the columns to have first letter in column name to avoid duplicates
def RenameCols(df, letter):
    for col in df.columns:
        if col in ['business_id', 'user_id']:
            continue
        elif col in ['type', 'neighborhoods', 'state']:
            df.drop(col, axis=1, inplace=True)
        else:
            df.rename(columns={col: letter+'_'+col}, inplace=True)

    return df

# fix all cities that are mispelled
def CityFix(df):
    for l, city in enumerate(df['city']):
        if city == 'Fountain Hls':
            df.loc[l, 'city'] = 'Fountain Hills'
        elif city == 'Glendale Az':
            df.loc[l, 'city'] = 'Glendale'
        elif city == 'Good Year':
            df.loc[l, 'city'] = 'Goodyear'
        elif city == 'Pheonix':
            df.loc[l, 'city'] = 'Phoenix'
        elif city == 'Scottsdale ':
            df.loc[l, 'city'] = 'Scottsdale'
    return df


def CategoryDummies(df):
    # Extract list of business categories - 508 total
    bus_categories = [item for sublist in list(df['b_categories']) for item in sublist]
    bus_categories = sorted(list(set(bus_categories)))

    # Create dummy variable for each category
    # Remove all spaces and characters in the name of category
    # Rename with b_categories
    for cat in bus_categories:
        df[cat] = np.asarray([1 if cat in row else 0 for row in df['b_categories']])
        cat2 = re.sub(u'[ &/-]', u'', cat)
        df.rename(columns={cat: 'b_categories_'+cat2}, inplace=True)

    return df

def excludeTesting(train_set, test_set):

    # Calculate number of reviews, total stars awarded, total votes (useful, funny, cool)
    # for each user in the test set
    test_set.loc[:, 'review_count'] = 1
    test_users = test_set.groupby('user_id', as_index=False).aggregate({'review_count': np.sum,
                                                                        'r_stars': np.sum,
                                                                        'r_votes_useful': np.sum,
                                                                        'r_votes_cool': np.sum,
                                                                        'r_votes_funny': np.sum})
    # Rename columns to distinguish from training set during merge
    test_users.rename(columns={'review_count': 'test_u_review_count',
                               'r_stars': 'test_u_review_stars',
                               'r_votes_useful': 'test_u_votes_useful',
                               'r_votes_cool': 'test_u_votes_cool',
                               'r_votes_funny': 'test_u_votes_funny'}, inplace=True)

    # Calculate number of reviews and total stars received for each business in the test data
    test_bus = test_set.groupby('business_id', as_index=False).aggregate({'review_count': np.sum,
                                                                          'r_stars': np.sum})
    # Rename columns to distinguish from training set during merge
    test_bus.rename(columns={'review_count': 'test_b_review_count',
                             'r_stars': 'test_b_review_stars'}, inplace=True)

    ## Spot tests
    # For testing with one specific user
    # train_set = train_set.loc[train_set.user_id=='shkOSzUcN2hjIJpyufVS9w', :]

    # Merge aggregated test set information about users/businesses into training set
    train_set = pd.merge(train_set, test_users, on='user_id', how='left')
    train_set = pd.merge(train_set, test_bus, on='business_id', how='left')

    # Fill in the merged rows with no matches with 0s (will not be updated during computations)
    new_col = ['test_u_review_count', 'test_u_review_stars', 'test_u_votes_useful', 'test_u_votes_cool',
               'test_u_votes_funny', 'test_b_review_count', 'test_b_review_stars']
    for col in new_col:
        train_set[col].fillna(0, inplace=True)

    ## Spot tests
    # user_col = [col for col in train_set.columns if 'u_' in col]
    # bus_col = [col for col in train_set.columns if 'b_' in col]
    # print "printing specific user"
    # print train_set.loc[train_set['user_id'] == 'shkOSzUcN2hjIJpyufVS9w', user_col]
    # print "printing specific business"
    # print train_set.loc[train_set['business_id'] == '-8wyZkzfBmCFkMwCGcR4PQ', bus_col]

    ## Correct the user-level information in the training set by removing test set reviews
    # new stars = ((average stars)*(total reviews) - (total test set stars)) / (# test set reviews)
    train_set['u_average_stars'] = (((train_set['u_average_stars']*train_set['u_review_count']) -
                                    train_set['test_u_review_stars']) /
                                    (train_set['u_review_count'] - train_set['test_u_review_count']))
    # number of training reviews = (total reviews) - (number of test reviews)
    train_set['u_review_count'] = train_set['u_review_count'] - train_set['test_u_review_count']
    # number of training votes = (total votes) - (number of test votes)
    train_set['u_votes_useful'] = train_set['u_votes_useful'] - train_set['test_u_votes_useful']
    train_set['u_votes_cool'] = train_set['u_votes_cool'] - train_set['test_u_votes_cool']
    train_set['u_votes_funny'] = train_set['u_votes_funny'] - train_set['test_u_votes_funny']

    ## Correct the business-level information in the training set by removing test set reviews
    # new stars = ((average stars)*(total reviews) - (total test set stars)) / (# test set reviews)
    train_set['b_stars'] = (((train_set['b_stars']*train_set['b_review_count']) - train_set['test_b_review_stars']) /
                            (train_set['b_review_count'] - train_set['test_b_review_count']))
    # number of training reviews = (total reviews) - (number of test reviews)
    train_set['b_review_count'] = train_set['b_review_count'] - train_set['test_b_review_count']

    # Drop test set information
    train_set.drop(new_col, axis=1, inplace=True)

    ## Spot tests
    # print "printing specific user"
    # print train_set.loc[train_set['user_id'] == 'shkOSzUcN2hjIJpyufVS9w', user_col]
    # print "printing specific business"
    # print train_set.loc[train_set['business_id'] == '-8wyZkzfBmCFkMwCGcR4PQ', bus_col]

    # Return updated test set
    return train_set


# shuffle the rows to randomize the data
def shuffle(df):
    random.seed(7)
    index = list(df.index)
    random.shuffle(index)
    df = df.ix[index]
    df = df.reset_index()
    return df

if __name__ == "__main__":
    # set the location of the data sets
    bus_data = "yelp_training_set/yelp_training_set_business.json"
    rev_data = "yelp_training_set/yelp_training_set_review.json"
    usr_data = "yelp_training_set/yelp_training_set_user.json"
    chk_data = "yelp_training_set/yelp_training_set_checkin.json"

    # open and rename all of the data sets
    bus = RenameCols(CityFix(OpenFile(bus_data)), 'b')
    rev = RenameCols(OpenFile(rev_data), 'r')
    usr = RenameCols(OpenFile(usr_data), 'u')
    chk = OpenFile(chk_data)

    # sum the total number of check-ins
    chk['b_sum_checkins'] = 0
    for b, row in enumerate(chk['checkin_info']):
        chk.loc[b, 'b_sum_checkins'] = sum(row.values())

    # only need business_id and total check-ins
    chk = chk[['business_id', 'b_sum_checkins']]

    # extract cool/funny/useful votes for users
    usr_votes = pd.DataFrame(list(usr['u_votes']), columns=['cool', 'funny', 'useful'])
    usr_votes = RenameCols(usr_votes, "u_votes")
    usr = pd.concat([usr, usr_votes], axis=1)

    # extract cool/funny/useful votes for reviews
    rev_votes = pd.DataFrame(list(rev['r_votes']), columns=['cool', 'funny', 'useful'])
    rev_votes = RenameCols(rev_votes, "r_votes")
    rev = pd.concat([rev, rev_votes], axis=1)

    # merge all of the datasets together on user and business ids
    full = pd.merge(rev, usr, on='user_id', how='left')
    print "Merged reviews & user datasets"
    full = pd.merge(full, bus, on='business_id', how='left')
    print "Merged in business data"
    full = pd.merge(full, chk, on='business_id', how='left')
    print "Merged in checkin data"

    # shuffle the indices of the dataset
    full_shuffle = shuffle(full)

    # split the data into training and test sets based on 80/20
    split = int(len(full)*0.8)
    training = full_shuffle[:split]
    test = full_shuffle[split:]
    print "Split the data"

    # remove any review information that are in the test set
    training = excludeTesting(training, test)
    print "Excluded any review information about test set"

    # create dummy variables for business categories
    training = CategoryDummies(training)
    test = CategoryDummies(test)
    print "Created category dummies"

    print "Exporting to CSV"
    # export to csv
    training.to_csv('yelp_training.csv', index=False, encoding='utf-8')
    test.to_csv('yelp_test.csv', index=False, encoding='utf-8')

    # get just the reviews text from training data
    text = training.loc[:, ['user_id', 'r_text']]
    text.to_csv('yelp_review_text.csv', index=False, encoding='utf-8')
    print "Finished"
