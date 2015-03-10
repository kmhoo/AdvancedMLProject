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

def get_reviews(fname):
    """
    get review text from the data set
    :param fname: file name of data set; expecting csv
    :return: pandas dataframe of text reviews (strings)
    """
    try:
        with open(fname, "rb") as infile:
            df = pd.DataFrame.from_csv(infile, header=0, index_col=False)
            print "# of user reviews: ", len(df)
            # drop any review entries that are blank
            df = df.dropna()
            # remove all newline characters from each entry
            df['r_text'] = df['r_text'].str.replace('\n', ' ')
            print "... after dropping NAs: ", len(df)
        return df
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
            print "# of businesses: ", len(df)
            # drop any review entries that are blank
            df = df.dropna()

            # remove all newline characters from each entry
            df['r_text'] = df['r_text'].str.replace('\n', ' ')
            print "... after dropping NAs: ", len(df)
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


def ngram_processing(trainText, testText):
    """
    Function to create array of data for processing in an algorithm.
    :param userIDs: list of users, ordered alphabetically (per 'sorted' corpus df)
    :param corpus: list of reviews by each user
    :param userReviewCounts: dictionary of {user_id : inverse review frequency} to weigh frequent reviewers less
    :return: pandas dataframe, where rows=users, columns=ngram features
    """
    dataSet = []

    print "Starting text feature extraction "

    # process review data from training data
    reviews = get_reviews(trainText)
    userIDlist, corpus, userReviewCounts = create_corpus(reviews)

    testReviews = get_reviews(testText)
    testUsrIDlist, testCorpus, testUsrRevCounts = create_corpus(testReviews)

    # create tfidf transformer object
    transformer = TfidfTransformer()

    # calculate tf for bigram/trigram features; set minimum document frequency to 1000
    ngram_vectorizer = CountVectorizer(ngram_range=(2,3), token_pattern=r'\b\w+\b', min_df=1000)

    # get counts of the bigrams across documents; create an array for
    print "fitting/vectorizing ngram features (this may take awhile)"
    text_features = ngram_vectorizer.fit_transform(corpus).toarray()

    # get feature names and their indices via 'vocabulary_'; result is a dictionary
    feature_names = ngram_vectorizer.get_feature_names()
    # print feature_names

    # get vectorized features from the test data set
    print "Vectorizing test reviews"
    testNgram_vectors = CountVectorizer(token_pattern=r'\b\w+\b', vocabulary=ngram_vectorizer.vocabulary_)

    testNgram_features = testNgram_vectors.fit_transform(testCorpus).toarray()

    # calculate tfidf scores for the ngram features; results in sparse matrix
    tfidf = transformer.fit_transform(text_features).toarray()

    # apply review frequency normalization for each user's Ngram tfidfs
    n = 0
    for user in userIDlist:
        # create new array of the tfidf normalized values for each row (user) in the tfidf array
        dataSet.append([userReviewCounts[user]* elem for elem in tfidf[n]])
        # reviewWeights.append(userReviewCounts[user])
        n += 1

    dataSet = np.array(dataSet)

    # print dataSet[:5]


    return dataSet


def create_corpus(reviews):
    """
    Function that takes in a dataframe or array of data and collapses the reviews into single strings by user_id
    :param df: data frame of user_ids and reviews
    :return:
    """
    corpus = []
    userIDlist = []
    userReviews = OrderedDict()
    userReviews2 = OrderedDict()

    print "Starting corpus generation"
    # get count of reviews by userid
    userReviewCounts = Counter(list(reviews['user_id']))

    # calculate weights for each user: 1/(# reviews by user)
    for k, v in userReviewCounts.items():
        userReviewCounts[k] = 1/float(v)

    # aggregate reviews by user ("document"); each user should now have a list of their reviews
    userReviewLists = reviews.groupby('user_id')['r_text'].apply(list)       # output is a pd.Series sorted by user_id

    # recast user review lists as dictionary values
    for usr, rlist in userReviewLists.iteritems():
        userReviews[usr] = rlist

    # print userReviewLists[:5]
    # create user_id to sparse-matrix-index hash table

    # create a list of "cleaned-up" strings for ngram vectorizer processing
    # join all the review strings together, using a ridiculous stop word (to exclude as a feature) to signify the
    # break between reviews (so that the last word of one review does not form a bigram with the first word of another)
    for usr, rlists in userReviews.iteritems():
        userReviews2[usr] = " hoobakerokamoto ".join(rlists)
    print "User review lists combined"

    # create a corpus of all reviews; each user's reviews is one document within the corpus
    for usr, rString in userReviews2.iteritems():
        corpus.append(text_preprocessing(rString))
        userIDlist.append(usr)

    # print userIDlist[:5]

    print "user_id list, corpus, and user review counts generated"
    return userIDlist, corpus, userReviewCounts



if __name__=="__main__":
    trainText = "yelp_review_text.csv"
    testText = "yelp_review_text_test.csv"

    trainingData = "training_2.csv"
    testData = "test_2.csv"


    trainTextFeatures = ngram_processing(trainText, testText)
    # reviews = get_reviews(trainText)

    # return corpus of data
    # userIDs, corpus = create_corpus(reviews)
    # ngram_processing(userIDs, corpus)

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