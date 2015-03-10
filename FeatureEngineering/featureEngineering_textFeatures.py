__author__ = 'jrbaker'
"""featureEngineering_textFeatures.py - script to generate n-gram features for user rating predictions
"""

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import nltk, string, csv, itertools, re
import pandas as pd
import numpy as np
from nltk.stem import wordnet
from nltk.tokenize import RegexpTokenizer, punkt, WordPunctTokenizer
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

    # exclude words with numerics
    tokens = [word for word in reviewTokens if re.search(r'[a-zA-Z]', word) is not None]

    # join all reviews together into one string
    tokens = " ".join(tokens)

    # print tokens
    return tokens


def ngram_processing(userIDs, corpus, test_data=False):
    """
    Function to create array of data for processing in an algorithm.
    :param test_data: flag to indicate if the incoming data set is training or testing data;
                      if test_data=True, then we only keep the ngram columns that exist in
                      training, and exclude the rest
    :return: numpy array of data; rows=users, columns=ngram features
    """
    dataSet = []
    userIDlist = userIDs

    # create tfidf transformer object
    transformer = TfidfTransformer()

    # calculate tf for bigram/trigram features; set minimum document frequency to 1000
    ngram_vectorizer = CountVectorizer(ngram_range=(2,3), token_pattern=r'\b\w+\b', min_df=1000)

    # get counts of the bigrams across documents; create an array for
    print "fitting/vectorizing ngram features (this may take awhile)"
    text_features = ngram_vectorizer.fit_transform(corpus).toarray()

    print len(text_features)

    print text_features[-5]

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

def create_corpus(df):
    """
    Function that takes in a dataframe or array of data and collapses the reviews into single strings by user_id
    :param df: data frame of user_ids and reviews
    :return:
    """
    corpus = []
    userReviews = OrderedDict()
    userReviews2 = OrderedDict()
    userIDlist = []
    reviewWeights = []

    # calculate weights for each user: 1/(# reviews by user)
    for k, v in userReviewCounts.items():
        userReviewCounts[k] = 1/float(v)

    # aggregate reviews by user ("document"); each user should now have a list of their reviews
    userReviewLists = reviews.groupby('user_id')['r_text'].apply(list)       # output is a pd.Series sorted by user_id

    # recast user review lists as dictionary values
    for usr, rlist in userReviewLists.iteritems():
        userReviews[usr] = rlist

    print userReviewLists[:5]
    # create user_id to sparse-matrix-index hash table

    # create a list of "cleaned-up" strings for bigram vectorizer processing
    # join all the review strings together, using a ridiculous stop word (to exclude as a feature)
    for usr, rlists in userReviews.iteritems():
        userReviews2[usr] = " hoobakerokamoto ".join(rlists)
    print "user review lists joined"

    # create a corpus of all reviews; each user's reviews is one document within the corpus
    for usr, rString in userReviews2.iteritems():
        corpus.append(text_preprocessing(rString))
        userIDlist.append(usr)

    print userIDlist[:5]

    print "corpus created"
    return userIDlist, corpus



if __name__=="__main__":
    path = "text_analysis/yelp_review_text.csv"
    train = "yelp_training.csv"
    testData = ""


    print "starting preprocess"
    reviews = get_reviews(path)
    # get count of reviews by userid
    userReviewCounts = Counter(list(reviews['user_id']))

    # return corpus of data
    userIDs, corpus = create_corpus(reviews)
    ngram_processing(userIDs, corpus)

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