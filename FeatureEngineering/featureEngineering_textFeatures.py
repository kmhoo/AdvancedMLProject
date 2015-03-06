__author__ = 'jrbaker'
"""featureEngineering_textFeatures.py - script to generate n-gram features for user rating predictions
"""

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import nltk, string, csv, itertools
import pandas as pd
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
    # reviewsList[:] = [str(s).decode('utf-8') if str(s) not in (None, 'nan', '') else '' for s in reviewsList]

    # print "reviewsList processed: ", reviewsList

    # group the list of strings together and remove stopwords
    reviewsList[:] = list(itertools.chain(*reviewsList))

    # print "iter-chained"

    reviewsList = [word for word in reviewsList if word not in stopwords]

    # join all reviews together into one string
    tokens = nltk.word_tokenize(" ".join(reviewsList))

    return tokens


if __name__=="__main__":
    path = "text_analysis/yelp_review_text.csv"
    corpus = []

    reviews = get_reviews(path)
    # [u'user_id', u'r_text']

    print "starting preprocess"
    print reviews.ix[1,:]


    # get count of reviews by userid
    userReviewCounts = Counter(list(reviews['user_id']))
    # print len(userReviewCounts)

    print userReviewCounts['BRNVGPDi58XPDyxfxX39sg']
    bigram_vectorizer = CountVectorizer(ngram_range=(2,3), token_pattern=r'\b\w+\b', min_df=1)
    # create a list of "cleaned-up" strings for bigram vectorizer processing

    # print reviews.dtypes
    # 1. aggregate reviews by user ("document")
    userReviews = reviews.groupby('user_id')['r_text'].apply(list)

    print len(userReviews['BRNVGPDi58XPDyxfxX39sg'])
    # 2. calculate tf-idf for bigram/trigram features

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





