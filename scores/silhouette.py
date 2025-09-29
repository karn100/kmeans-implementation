import numpy as np
from scratch import Kmeans_scratch
from typing import List,Optional


def silhouette_score(X:np.ndarray) -> float:
    kmeans = Kmeans_scratch()
    kmeans.fit(X)
    if kmeans.labels is None:
        raise ValueError("Model has not been fitted yet")
    n_samples = X.shape[0]
    scores = []
    for i in range(n_samples):
        same_clusters = X[kmeans.labels == kmeans.labels[i]]
        other_clusters = [X[kmeans.labels == j] for j in set(kmeans.labels) if j != kmeans.labels[i]]
        
        a = np.mean(np.linalg.norm(same_clusters - X[i],axis = 1)) if len(same_clusters) > 1 else 0
        b = np.min([np.mean(np.linalg.norm(other - X[i],axis = 1))for other in other_clusters]) if other_clusters else 0
        