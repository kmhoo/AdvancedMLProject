__author__ = 'kaileyhoo'

import pandas as pd
from data_processing import updateColumns, roughImpute
from sklearn.cluster import KMeans

training = pd.read_csv("../yelp_training.csv")
training = updateColumns(training)
training = roughImpute(training)

X = training.loc[:, ['u_votes_cool', 'u_votes_funny', 'u_votes_useful', 'u_review_count']]

clusters = [2, 3, 4, 5, 8, 10, 15, 20, 25, 50]
init = [10, 30, 50, 100]

cluster_list = []
for cl in clusters:
    iter = {}
    iter['num_of_clusters'] = cl
    model = KMeans(n_clusters=cl)
    model.fit(X)
    iter['labels'] = model.labels_
    iter['centroids'] = model.cluster_centers_
    cluster_list.append(iter)
print cluster_list