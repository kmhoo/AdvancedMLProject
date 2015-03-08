__author__ = 'jrbaker'
"""featureEngineering_textFeatures.py - script to generate n-gram features for user rating predictions
"""

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import nltk, string, csv, itertools, re
import pandas as pd
import numpy as np
from nltk.stem import wordnet
from nltk.tokenize import RegexpTokenizer, punkt, WordPunctTokenizer
from collections import Counter

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

    # print review

    # print "reviewsList: ", list(textIter)


    # clean up text to remove punctuation and empty reviews
    reviewTokens = [review.translate(None, string.punctuation).lower().split()]

    # print "reviewTokens processed: ", reviewTokens

    # group the list of strings together and remove stopwords
    reviewTokens = list(itertools.chain(*reviewTokens))

    # print "iter-chained"

    reviewTokens = [word for word in reviewTokens if word not in stopwords]

    # print "stopwords removed: ", reviewTokens

    tokens = [word for word in reviewTokens if re.search(r'[a-zA-Z]', word) is not None]

    # join all reviews together into one string
    tokens = " ".join(tokens)

    # print tokens
    return tokens


if __name__=="__main__":
    path = "text_analysis/yelp_review_text.csv"
    train = "yelp_training.csv"
    corpus = []
    userReviews = {}
    userReviews2 = {}

    print "starting preprocess"
    reviews = get_reviews(path)
    # print reviews

    # print reviews[reviews["user_id"].isin(['RcfkeXHjYWCpq6NyVVmGJg']) ]

    # 41,000 users

    # get business_ids and their reviews
    bReviews = get_b_reviews(train)

    # print bReviews

    bReviewCounts = Counter(list(bReviews['business_id']))
    # print len(bReviewCounts)    # 11,508

    # aggregate reviews by business ("document")
    businessReviewLists = bReviews.groupby('business_id')['r_text'].apply(list)
    # print businessReviewLists['RcfkeXHjYWCpq6NyVVmGJg']

    # print businessReviewLists

    # get count of reviews by userid
    userReviewCounts = Counter(list(reviews['user_id']))
    # print userReviewCounts

    # 3. apply weight to each user: 1/(# reviews by user)
    for k, v in userReviewCounts.items():
        userReviewCounts[k] = 1/float(v)

    # print userReviewCounts['fczQCSmaWF78toLEmb0Zsw']         # test case

    # 1. aggregate reviews by user ("document"); each user should now have a list of their reviews
    userReviewLists = reviews.groupby('user_id')['r_text'].apply(list)

    # recast user review lists as dictionary values
    for i, v in userReviewLists.iteritems():
        userReviews[i] = v

    # print "fczQCSmaWF78toLEmb0Zsw", len(userReviews["fczQCSmaWF78toLEmb0Zsw"]), userReviews["fczQCSmaWF78toLEmb0Zsw"]

    # create a list of "cleaned-up" strings for bigram vectorizer processing

    # join all the review strings together, using a ridiculous stop word (to exclude as a feature)
    for usr, rlists in userReviews.iteritems():
        # print usr, rlists
        userReviews2[usr] = " hoobakerokamoto ".join(rlists)
        # print userReviews2
    print "lists joined"

    # create a corpus of all reviews; each user's reviews is one document within the corpus
    for usr, rString in userReviews2.iteritems():
        # userReviews2[usr] = text_preprocessing(rString)
        corpus.append(text_preprocessing(rString))

    print "text_preprocessing done...?"
    # print "fczQCSmaWF78toLEmb0Zsw", userReviews2["fczQCSmaWF78toLEmb0Zsw"]
    # 3. calculate tf-idf for bigram/trigram features

    ngram_vectorizer = CountVectorizer(ngram_range=(2,3), token_pattern=r'\b\w+\b', min_df=1)

    transformer = TfidfTransformer()
    print "corpus: "

    # get counts of the bigrams across documents;
    text_features = ngram_vectorizer.fit_transform(corpus)

    print "feature len: ", len(text_features)

    # print text_features.todense()[:10]

    print text_features[:5]

    tfidf = transformer.fit_transform(text_features)
    print tfidf

    print ngram_vectorizer.get_feature_names()[:5]
    print ngram_vectorizer.inverse_transform(tfidf)[:5]
    tfidfArray = tfidf.toarray()

    print len(tfidfArray)                             #  rows
    print len(ngram_vectorizer.get_feature_names())   # 116,006 feature names


