__author__ = 'jrbaker'
"""
featureEngineering_textFeatures.py - script to generate n-gram features for user rating predictions
'ngram_processing' is the main function that generates ngram features for the training data
"""

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import nltk, string, csv, itertools, re
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from data_cleaning import shuffle

def get_reviews(fname, training):
    """
    get review text from the data set
    :param fname: file name of data set; expecting csv
    :return: pandas dataframe of text reviews (strings)
    """
    train_users = training.loc[:, ['user_id', 'business_id', 'u_review_count_update']]
    # df = pd.read_csv(fname, header=0, index_col=False)
    try:
        with open(fname, "rb") as infile:
            df = pd.DataFrame.from_csv(infile, header=0, index_col=False)
            # print "# of user reviews: ", len(df)

            # merge with training data to get only reviews we have
            new_df = pd.merge(df, train_users, on=['user_id', 'business_id'])
            # drop any review entries that are blank
            new_df = new_df.dropna()
            # remove all newline characters from each entry
            new_df['r_text'] = new_df['r_text'].str.replace('\n', ' ')
            # print "... after dropping NAs: ", len(new_df)
        return new_df
    except:
        raise IOError


def get_b_reviews(fname):
    """
    get business review text from the training data set
    :param fname: file name of data set; expecting csv
    :return: pandas dataframe of text reviews (strings)
    """
    try:
        with open(fname, "rb") as infile:
            df = pd.DataFrame.from_csv(infile, header=0, index_col=False)
            df = df[['business_id', 'r_text']]
            # print "# of businesses: ", len(df)
            # drop any review entries that are blank
            df = df.dropna()

            # remove all newline characters from each entry
            df['r_text'] = df['r_text'].str.replace('\n', ' ')
            # print "... after dropping NAs: ", len(df)
        return df
    except:
        raise IOError


def text_preprocessing(textString):
    """
    remove stopwords, punctuation, etc., and stem/tokenize text strings
    :param textString: string of review text
    :return: list of tokens, grouped by document
    """
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    # addstopwords = [None, 'nan', '']
    stopwords.extend(['hoobakerokamoto'])
    review = textString

    # clean up text to remove punctuation and empty reviews
    reviewTokens = [review.translate(None, string.punctuation).lower().split()]

    # group the list of strings together and remove stopwords
    reviewTokens = list(itertools.chain(*reviewTokens))

    # clear stopwords
    reviewTokens = [word for word in reviewTokens if word not in stopwords]

    # exclude words that are only numerics
    tokens = [word for word in reviewTokens if re.search(r'[a-zA-Z]', word) is not None]

    # join all reviews together into one string
    tokens = " ".join(tokens)

    # print tokens
    return tokens


def create_corpus(reviews):
    """
    Function that takes in a dataframe or array of data and collapses the reviews into single strings by user_id
    :param reviews: data frame of user_ids and reviews
    :return: userIDlist - list of the user_ids, sorted alphabetically ascending
             corpus - array of combined reviews by user
             userReviewCounts - dictionary of {user_id : inverse review frequency} to weigh frequent reviewers less
    """
    corpus = []
    userIDlist = []
    userReviews = OrderedDict()
    userReviews2 = OrderedDict()

    # print "Starting corpus generation"
    # get count of reviews by userid
    userReviewCounts = Counter(list(reviews['user_id']))

    # calculate weights for each user: 1/(# reviews by user)
    for k, v in userReviewCounts.items():
        userReviewCounts[k] = 1/float(v)

    # aggregate reviews by user ("document"); each user should now have a list of their reviews
    # output is a pd.Series sorted by user_id
    userReviewLists = reviews.groupby('user_id')['r_text'].apply(list)

    # # attempt to remove current review
    # # convert into pandas dataframe
    # list_of_reviews = pd.DataFrame(userReviewLists).reset_index()
    #
    # # merge list of reviews back into dataframe
    # reviews = pd.merge(reviews, list_of_reviews, on='user_id', how='left')
    # reviews.rename(columns={'r_text_x': 'r_text', 'r_text_y': 'r_text_list'}, inplace=True)
    #
    # reviews['r_text_list_update'] = np.nan
    #
    # for idx, row in reviews.iterrows():
    #     print idx
    #     current_review = row['r_text']
    #     print current_review
    #     review_list = row['r_text_list']
    #     print review_list
    #     review_list.remove(current_review)
    #     print review_list
    #     reviews.loc[idx, 'r_text_list_update'] = review_list
    #     print row['r_text_list_update']
    # print 'UPDATE', len(reviews['r_text_list_update'][130])

    # recast user review lists as dictionary values
    for usr, rlist in userReviewLists.iteritems():
        userReviews[usr] = rlist

    # print userReviewLists[:5]

    # create a list of "cleaned-up" strings for ngram vectorizer processing
    # join all the review strings together, using a ridiculous stop word (to exclude as a feature) to signify the
    # break between reviews (so that the last word of one review does not form a bigram with the first word of another)
    for usr, rlists in userReviews.iteritems():
        userReviews2[usr] = " hoobakerokamoto ".join(rlists)
    # print "User review lists combined"

    # create a corpus of all reviews; each user's reviews is one document within the corpus
    for usr, rString in userReviews2.iteritems():
        corpus.append(text_preprocessing(rString))
        userIDlist.append(usr)

    # print userIDlist[:5]

    # print "user_id list, corpus, and user review counts generated"
    # print type(userIDlist), type(corpus), type(userReviewCounts)
    return userIDlist, corpus, userReviewCounts


def ngram_processing(train_df, test_df):
    """
    Function to create array of data for processing in an algorithm.
    :param trainText, testText: training and test data; must include fields 'user_id' and 'r_text'
    :return: pandas dataframe, where rows=users, columns=ngram features
    """
    dataSet = []

    # read in the text files
    trainText = "../yelp_review_text.csv"

    #subset for just reviews in training df (param)

    # print "Starting text feature extraction "

    # process review data from training data
    reviews = get_reviews(trainText, train_df)
    userIDlist, corpus, userReviewCounts = create_corpus(reviews)

    # create frequency count vectorizer; set minimum document frequency to 1000
    ngramVectorizer = CountVectorizer(ngram_range=(2, 3), token_pattern=r'\b\w+\b', min_df=500)

    # calculate frequencies for bigram/trigram features
    # print "Fitting/vectorizing ngram features (this may take awhile)"
    textFeatures = ngramVectorizer.fit_transform(corpus).toarray()

    # get feature names and their indices via 'vocabulary_'; result is a dictionary
    feature_names = ngramVectorizer.get_feature_names()
    # print "# features trained:", len(feature_names)


    # calculate tfidf scores for the ngram features; results in sparse matrix
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(textFeatures).toarray()

    # apply review frequency normalization for each user's Ngram tfidfs
    n = 0
    for user in userIDlist:
        # create new array of the tfidf normalized values for each row (user) in the tfidf array
        dataSet.append([userReviewCounts[user] * elem for elem in tfidf[n]])
        n += 1

    # convert dataset to pandas dataframe
    txt_tfidf = pd.DataFrame(dataSet, columns=feature_names)
    txt_tfidf['user_id'] = userIDlist

    #merging tfidif scores onto the training and test sets
    train_update = pd.merge(train_df, txt_tfidf, on='user_id', how='left')
    train_update.fillna(train_update.mean(), inplace=True)

    test_update = pd.merge(test_df, txt_tfidf, on='user_id', how='left')
    test_update.fillna(test_update.mean(), inplace=True)

    return train_update, test_update


if __name__=="__main__":
    trainText = "yelp_review_text.csv"
    testText = "yelp_review_text_test.csv"

    trainingData = pd.read_csv("../training_2.csv")
    trainingData = shuffle(trainingData)
    trainingData = trainingData[:int(.5*len(trainingData))]
    testData = pd.read_csv("../testing_2.csv")

    ngram_processing(trainingData, testData)


    ####################################
    # BUSINESS REVIEW COUNT SECTION
    # get business_ids and their reviews
    # bReviews = get_b_reviews(train)
    # create counts of the businesses and their reviews
    # bReviewCounts = Counter(list(bReviews['business_id']))

    # aggregate reviews by business ("document")
    # businessReviewLists = bReviews.groupby('business_id')['r_text'].apply(list)
    # END BUSINESS REVIEW COUNT SECTION
    ####################################

    # input: training nparray, test nparray, user_array
    # 1. turn training/test into pd.DFs
    # 2. set: users_train['user_id'] = users

    # output: 43k x 163 features for appending to other df
    # drop user_id