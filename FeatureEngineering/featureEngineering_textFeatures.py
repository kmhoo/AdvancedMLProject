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
            # print len(df)
            # drop any review entries that are blank
            df.dropna()

            # remove all newline characters from each entry
            df['r_text'] = df['r_text'].str.replace('\n', ' ')
            # print len(df)
        return df
    except:
        raise IOError


def text_preprocessing(textIter):
    """
    remove stopwords, punctuation, etc., and stem/tokenize text strings
    :param textIter: iterable of user_id, text (e.g. list, dataframe, etc.)
    :return: list of tokens, grouped by document
    """
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    # addstopwords = [None, 'nan', '']
    # stopwords.extend(addstopwords)

    reviewsList = textIter

    # print "reviewsList: ", list(textIter)

    # clean up text to remove punctuation and empty reviews
    reviewsList[:] = [s.translate(None, string.punctuation).lower().split()
                      if str(s) not in (None, 'nan', '') else '' for s in reviewsList]

    # print "reviewsList processed: ", reviewsList

    # group the list of strings together and remove stopwords
    reviewsList[:] = list(itertools.chain(*reviewsList))

    # print "iter-chained"

    reviewsList = [word for word in reviewsList if word not in stopwords]
    reviewsList = [word for word in reviewsList if re.search(r'[a-zA-Z]', word) is not None]

    # join all reviews together into one string
    tokens = nltk.word_tokenize(" ".join(reviewsList))

    return tokens


if __name__=="__main__":
    path = "text_analysis/yelp_review_text.csv"
    corpus = []
    userReviews = {}
    userReviews2 = {}

    reviews = get_reviews(path)
    print "starting preprocess"

    # get count of reviews by userid
    userReviewCounts = Counter(list(reviews['user_id']))
    # print len(userReviewCounts)

    # print userReviewCounts['BRNVGPDi58XPDyxfxX39sg']         # test case
    # create a list of "cleaned-up" strings for bigram vectorizer processing

    # 1. aggregate reviews by user ("document")
    userReviewLists = reviews.groupby('user_id')['r_text'].apply(list)

    for i, v in userReviewLists.iteritems():
        userReviews[i] = v

    print userReviews['BRNVGPDi58XPDyxfxX39sg']

    # 2. combine review strings by user
    for usr, rlists in userReviews.iteritems():
        userReviews2[usr] = text_preprocessing(rlists)

    print userReviews2['BRNVGPDi58XPDyxfxX39sg']
    # 3. calculate tf-idf for bigram/trigram features

    bigram_vectorizer = CountVectorizer(ngram_range=(2,3), token_pattern=r'\b\w+\b', min_df=1)

    # 3. apply weight to each user: 1/(# reviews by user)


    # for docString in docStrings:
        # print " ".join(docString)
        # corpus.append(" ".join(docString))
        #
        # transformer = TfidfTransformer()
        #
        # tfidf = transformer.fit_transform(X_2)
        #
        # print bigram_vectorizer.get_feature_names()
        # print tfidf.toarray()





