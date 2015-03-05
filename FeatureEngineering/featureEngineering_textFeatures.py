__author__ = 'jrbaker'
"""featureEngineering_textFeatures.py - script to generate n-gram features for user rating predictions
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, string, csv, itertools
import pandas as pd
from nltk.stem import wordnet
from nltk.tokenize import RegexpTokenizer, punkt, WordPunctTokenizer

def get_reviews(fname):
    """
    get review text from the data set
    :param fname: file name of data set; expecting csv
    :return: pandas dataframe of text reviews (strings)
    """
    try:
        with open(fname, "rb") as infile:
            df = pd.DataFrame.from_csv(infile, header=0, index_col=False)
            print df.columns
            # drop any review entries that are blank
            df.dropna()

            # remove all newline characters from each entry
            df.replace({'\n' : ''}, inplace=True)
            # df.apply(lambda x: x.str.replace('\n', ''))

        return df
    except:
        raise IOError


def text_preprocessing(textIter):
    """
    remove stopwords, punctuation, etc., and stem/tokenize text strings
    :param textIter: iterable of text (e.g. list, dataframe, etc.)
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
    path = "yelp_review_text.csv"

    reviewCorpus = get_reviews(path)
    print "1: ", reviewCorpus.ix[2,:]

    reviewCorpus = list(reviewCorpus.values.flatten())
    print "starting preprocess"
    print "2: ", reviewCorpus[2]

    reviewTokens = text_preprocessing(reviewCorpus)

    # print type(reviewTokens)

