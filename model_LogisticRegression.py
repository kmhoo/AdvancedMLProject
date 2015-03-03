__author__ = 'kaileyhoo'

from data_processing import numpyArrays
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.learning_curve import validation_curve


def LogisticReg1(X, y):

    clf = LogisticRegression()
    train_scores, test_scores = validation_curve(clf, X, y, cv=5, scoring='accuracy')
    return train_scores, test_scores


if __name__ == "__main__":
    data, target = numpyArrays("yelp_training.csv")

    lg_train, lg_test = LogisticReg1(data, target)
    print lg_test

