__author__ = 'kaileyhoo'

import pandas as pd
from data_processing import updateColumns, roughImpute
from sklearn.cluster import KMeans

training = pd.read_csv("yelp_training.csv")
training = updateColumns(training)
training = roughImpute(training)

print training
columns = ['u_votes_cool', 'u_votes_funny', 'u_votes_useful', 'u_review_count', 'u_stars']
X = training.loc[:, columns]
print X

clusters = [2, 3, 4, 5, 8, 10, 15, 20, 25, 50]
init = [10, 30, 50, 100]

for cl in clusters:
    print cl
    model = KMeans(n_clusters=cl)
    model.fit(X)
    k_means_labels = model.labels_
    print k_means_labels
    print model.score(X)