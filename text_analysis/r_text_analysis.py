__author__ = 'jrbaker'
"""
r_text_analysis.py - processes yelp review text to retrieve N-gram frequencies

"""


import csv, re, itertools
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk, string
from nltk.stem import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, punkt, WordPunctTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def get_reviews(fname):
    """
    get review text from the data set
    :param fname: file name of data set; expecting csv
    :return: pandas dataframe of text reviews (strings)
    """
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)

    try:
        with open(fname, "rb") as infile:
            df = pd.DataFrame.from_csv(infile, header=0, index_col=False)
            # drop any review entries that are blank
            print "length of df: ", len(df)
            df = list(df['r_text'].dropna())
            print "... after removing NAs: ", len(df)
            # clean up text to remove punctuation and empty reviews
            reviewsList = [s.replace('\n', '').lower() for s in df]

        return reviewsList
    except:
        raise IOError


def stem_tokenize(str_use):
    """
    Takes a string and tokenizes it, stripping it of punctuation and stopwords. Returns a list of strings.
    """
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    stopwords.append('')
    addstopwords = ["in", "on", "of", "''"]
    stopwords.append(addstopwords)
    stemmer = wordnet.WordNetLemmatizer()
    tokenizer = punkt.PunktWordTokenizer()

    # removes stopwords and punctuation, then splits the string into a list of words
    token = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(str_use)
             if token.lower().strip(string.punctuation) not in stopwords]
    text = [word for word in token if re.search(r'[a-zA-Z]', word) is not None]
    stem = [stemmer.lemmatize(word) for word in text]
    # Returns a list of strings
    return stem


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

    # clean up text to remove punctuation and empty reviews
    reviewsList[:] = [s.translate(None, string.punctuation).lower().split()
                      if str(s) not in (None, 'nan', '') else '' for s in reviewsList]

    # group the list of strings together and remove stopwords
    reviewsList[:] = list(itertools.chain(*reviewsList))
    print "count of tokens before stopword removal: ", len(reviewsList)
    reviewsList = [word for word in reviewsList if word not in stopwords]
    reviewsList = [word for word in reviewsList if re.search(r'[a-zA-Z]', word) is not None]
    print "count of tokens after stopword removal: ", len(reviewsList)

    # join all reviews together into one string
    tokens = nltk.word_tokenize(" ".join(reviewsList))

    return tokens


def get_ngrams(tokens):
    """
    :param tokens: list of terms from all reviews
    :return: outputs unigram, bigram, and trigram files
    """

    # print tokens

    print "processing N-grams"
    # collect the bigrams and trigrams
    bigrams = nltk.bigrams(tokens)
    trigrams = nltk.trigrams(tokens)
    # cfd = nltk.ConditionalFreqDist(bigrams)

    unigrams = Counter(tokens)
    print "Total tokens (after stopword removal)"
    print "Unique unigrams: ", len(unigrams)
    print "unigram frequencies generated"
    bigramFdist = nltk.FreqDist(bigrams)
    print "bigram frequencies generated"
    trigramFdist = nltk.FreqDist(trigrams)
    print "trigram frequencies generated"

    # write out n-gram lists to file
    path1 = 'review_unigrams.csv'
    writer = csv.writer(open(path1, 'wb'))
    for term, val in unigrams.most_common():
        # print term, val
        writer.writerow([term, val])
    print "Unigrams done: ", str(path1)

    path2 = 'review_bigrams.csv'
    writer2 = csv.writer(open(path2, 'wb'))
    for term, val in bigramFdist.items():
        # print term, val
        writer2.writerow([term, val])
    print "Bigrams done: ", str(path2)


    path3 = 'review_trigrams.csv'
    writer3 = csv.writer(open(path3, 'wb'))
    for term, val in trigramFdist.items():
        # print term, val
        writer3.writerow([term, val])
    print "Trigrams done: ", str(path3)




if __name__=="__main__":
    # pull user review text from data
    x = get_reviews("yelp_review_text.csv")
    reviews = []

    # print len(x)
    # convert to list of strings; output is a list of sentences
    # reviewsList = list(x.values.flatten())
    reviewsList = x
    print "list of review sentences generated"
    # print reviewsList[:3]
    # get tokens for all reviews
    tokens = text_preprocessing(reviewsList)
    print "review text tokenized"

    # write out files of n-grams
    get_ngrams(tokens)
    print "All Ngram frequency files created"
