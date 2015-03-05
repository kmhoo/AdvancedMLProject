__author__ = 'griffin'


from data_cleaning import *
import re

# set the location of the data sets
bus_data = "yelp_test_set/yelp_test_set_business.json"
rev_data = "yelp_test_set/yelp_test_set_review.json"
usr_data = "yelp_test_set/yelp_test_set_user.json"
chk_data = "yelp_test_set/yelp_test_set_checkin.json"

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

# export to csv to do exploratory data analysis
full.to_csv('yelp_test.csv', index=False, encoding='utf-8')