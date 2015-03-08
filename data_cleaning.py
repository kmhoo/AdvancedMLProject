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
    # user_dict = {}
    # bus_dict = {}
    # # go through the test set to get all unique users and businesses
    # for te, te_row in test_set.iterrows():
    #     # sum all reviews, stars, and votes in the test set by user and business
    #     if te_row['user_id'] in user_dict.keys():
    #         user_dict[te_row['user_id']]['review_count'] += 1
    #         user_dict[te_row['user_id']]['review_stars'] += te_row['r_stars']
    #         user_dict[te_row['user_id']]['r_votes_cool'] += te_row['r_votes_cool']
    #         user_dict[te_row['user_id']]['r_votes_funny'] += te_row['r_votes_funny']
    #         user_dict[te_row['user_id']]['r_votes_useful'] += te_row['r_votes_useful']
    #     else:
    #         user_dict[te_row['user_id']] = {'review_count': 1,
    #                                         'review_stars': te_row['r_stars'],
    #                                         'r_votes_cool': te_row['r_votes_cool'],
    #                                         'r_votes_funny': te_row['r_votes_funny'],
    #                                         'r_votes_useful': te_row['r_votes_useful']}
    #     if te_row['business_id'] in bus_dict.keys():
    #         bus_dict[te_row['business_id']]['review_count'] += 1
    #         bus_dict[te_row['business_id']]['review_stars'] += te_row['r_stars']
    #     else:
    #         bus_dict[te_row['business_id']] = {'review_count': 1,
    #                                            'review_stars': te_row['r_stars']}

    # Faster computation? Using pandas aggregate functions
    # Calculate number of reviews, total stars awarded, total votes (useful, funny, cool)
    # for each user in the test set
    test_set.loc[:, 'review_count'] = 1
    test_users = test_set.groupby('user_id', as_index=False).aggregate({'review_count': np.sum,
                                                                        'r_stars': np.sum,
                                                                        'r_votes_useful': np.sum,
                                                                        'r_votes_cool': np.sum,
                                                                        'r_votes_funny': np.sum})
    test_users.rename(columns={'r_stars': 'review_stars'}, inplace=True)

    # Convert pandas data frame to dictionary (key=user_id)
    user_dict = test_users.set_index('user_id').T.to_dict('dict')
    # print user_dict

    # Calculate number of reviews and total stars received for each business in the test data
    test_bus = test_set.groupby('business_id', as_index=False).aggregate({'review_count': np.sum,
                                                                          'r_stars': np.sum})
    test_bus.rename(columns={'r_stars': 'review_stars'}, inplace=True)

    # Convert to dictionary (key=business_id)
    bus_dict = test_bus.set_index('business_id').T.to_dict('dict')
    # print bus_dict

    # For testing with one specific user
    # train_set = train_set.loc[train_set.user_id=='shkOSzUcN2hjIJpyufVS9w', :]

    # update the training data for businesses and users
    for tr, tr_row in train_set.iterrows():
        # update user information to exclude test reviews
        if tr_row['user_id'] in user_dict.keys():
            new_review_total = tr_row['u_review_count'] - user_dict[tr_row['user_id']]['review_count']
            train_set.loc[tr, 'u_average_stars'] = ((tr_row['u_average_stars'] * tr_row['u_review_count']) -
                                         user_dict[tr_row['user_id']]['review_stars']) / new_review_total
            train_set.loc[tr, 'u_review_count'] = new_review_total
            train_set.loc[tr, 'u_votes_funny'] = tr_row['u_votes_funny'] - user_dict[tr_row['user_id']]['r_votes_funny']
            train_set.loc[tr, 'u_votes_cool'] = tr_row['u_votes_cool'] - user_dict[tr_row['user_id']]['r_votes_cool']
            train_set.loc[tr, 'u_votes_useful'] = tr_row['u_votes_useful'] - user_dict[tr_row['user_id']]['r_votes_useful']

        # update business information to exclude test reviews
        if tr_row['business_id'] in bus_dict.keys():
            new_review_total_b = tr_row['b_review_count'] - bus_dict[tr_row['business_id']]['review_count']
            train_set.loc[tr, 'b_stars'] = ((tr_row['b_stars'] * tr_row['b_review_count']) -
                                 bus_dict[tr_row['business_id']]['review_stars']) / new_review_total_b
            train_set.loc[tr, 'b_review_count'] = new_review_total_b

    # print "printing specific user"
    # print train_set.loc[train_set['user_id']=='shkOSzUcN2hjIJpyufVS9w', ]
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

    # create dummy variables for business categories
    training = CategoryDummies(training)
    test = CategoryDummies(test)
    print "Created category dummies"

    # remove any review information that are in the test set
    training = excludeTesting(training, test)
    print "Excluded any review information about test set"

    print "Exporting to CSV"
    # export to csv
    training.to_csv('yelp_training.csv', index=False, encoding='utf-8')
    test.to_csv('yelp_test.csv', index=False, encoding='utf-8')

    # get just the reviews text from training data
    text = training.loc[:, ['user_id', 'r_text']]
    text.to_csv('yelp_review_text.csv', index=False, encoding='utf-8')
    print "Finished"
