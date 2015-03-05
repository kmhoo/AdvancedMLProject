__author__ = 'kaileyhoo'

import pandas as pd
import numpy as np
from data_processing import updateColumns, roughImpute
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt

training = pd.read_csv("../yelp_training.csv")
training = updateColumns(training)
training = roughImpute(training)

X = training.loc[:, ['u_votes_cool', 'u_votes_funny', 'u_votes_useful', 'u_review_count']]

clusters = [2, 3, 4, 5, 8, 10, 15, 20, 25, 50]

labels = []
centroids = []
for cl in clusters:
    iter = {}
    iter['num_of_clusters'] = cl
    model = KMeans(n_clusters=cl)
    model.fit(X)
    labels.append(model.labels_)
    centroids.append(model.cluster_centers_)

print len(labels)
print len(centroids)

k_euclid = [cdist(X, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]

# Total within-cluster sum of squares
wcss = [sum(d**2) for d in dist]
# The total sum of squares
tss = sum(pdist(X)**2)/X.shape[0]

# The between-cluster sum of squares
bss = tss - wcss

print bss

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(clusters, bss/tss*100, 'b*-')
ax.set_ylim((0, 100))
plt.grid(True)
plt.xlabel('n_clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Variance Explained vs. k')
plt.savefig("elbow_curve.png")