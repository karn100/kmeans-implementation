import numpy as np
from scratch import Kmeans_scratch
from typing import List,Optional


def silhouette_score(X:np.ndarray) -> float:
    kmeans = Kmeans_scratch()
    kmeans.fit(X)
    if kmeans.labels is None:
        raise ValueError("Model has not been fitted yet")
    n_samples = X.shape[0]
    
