__author__ = 'kaileyhoo'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score

training = pd.read_csv("../training_init.csv")

users = training.loc[:, ['u_votes_funny', 'u_votes_cool', 'u_votes_useful', 'u_review_count']]
users_scaled = scale(np.array(users))

clusters = [2, 3, 4, 5, 8, 10, 15, 20, 30, 50]

labels = []
centroids = []
scores = []
for cl in clusters:
    model = KMeans(n_clusters=cl)
    model.fit(users_scaled)
    labels.append(model.labels_)
    print "Labels"
    centroids.append(model.cluster_centers_)
    print "Centroids"
    score = silhouette_score(users_scaled, model.labels_, metric="euclidean")
    scores.append(scores)
    print "Scores"

print scores
