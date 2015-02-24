__author__ = 'kaileyhoo'

import pandas as pd
import json

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
        else:
            df.rename(columns={col: letter+'_'+col}, inplace=True)
    return df

# set the location of the data sets
bus_data = "yelp_training_set/yelp_training_set_business.json"
rev_data = "yelp_training_set/yelp_training_set_review.json"
usr_data = "yelp_training_set/yelp_training_set_user.json"

# open and rename all of the data sets
bus = RenameCols(OpenFile(bus_data), 'b')
rev = RenameCols(OpenFile(rev_data), 'r')
usr = RenameCols(OpenFile(usr_data), 'u')

# get the check-in information for each business
chk_data = "yelp_training_set/yelp_training_set_checkin.json"
chk = OpenFile(chk_data)

# sum the total number of check-ins
chk['sum_checkins'] = 0
for b, row in enumerate(chk['checkin_info']):
    chk.loc[b, 'sum_checkins'] = sum(row.values())

# only need business_id and total check-ins
chk = chk[['business_id', 'sum_checkins']]

# merge all of the datasets together on user and business ids
full = pd.merge(rev, usr, on='user_id', how='left')
full = pd.merge(full, bus, on='business_id', how='left')
full = pd.merge(full, chk, on='business_id', how='left')

# export to csv to do exploratory data analysis
full.to_csv('yelp_training.csv', index=False, encoding='utf-8')








