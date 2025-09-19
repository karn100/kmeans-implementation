import numpy as np

class Kmeans_scratch:
    def __init__(self,k = 3,max_iters = 100,n_init = 10,tol = 1e-4,init = 'kmeans++',distance = 'euclidean',random_state = None,verbose = False):

        self.k = k
        self.max_iters = max_iters
        self.n_init = n_init
        self.tol = tol
        self.init = init
        self.distance = distance
        self.random_state = random_state
        self.verbose = verbose

        self.centroids = None
        self.labels = None
        self.inertia = None
        self.inertia_history = []
        
        