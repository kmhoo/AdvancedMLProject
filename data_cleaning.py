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

# shuffle the rows to randomize the data
def shuffle(df):
    index = list(df.index)
    random.shuffle(index)
    df = df.ix[index]
    df.reset_index()
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

    # Extract list of business categories - 508 total
    bus_categories = [item for sublist in list(bus['b_categories']) for item in sublist]
    bus_categories = sorted(list(set(bus_categories)))
    # Create dummy variable for each category
    # Note: must remove one dummy to avoid multicollinearity
    # Remove all spaces and characters in the name of category
    # Rename with b_categories
    for cat in bus_categories:
        bus[cat] = [1 if cat in row else 0 for row in bus['b_categories']]
        cat2 = re.sub(u'[ &/-]', u'', cat)
        bus.rename(columns={cat: 'b_categories_'+cat2}, inplace=True)

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

    # get just the reviews text from training data
    text = training['r_text']
    text.to_csv('yelp_review_text.csv', index=False, encoding='utf-8')

    # export to csv
    training.to_csv('yelp_training.csv', index=False, encoding='utf-8')
    test.to_csv('yelp_test.csv', index=False, encoding='utf-8')