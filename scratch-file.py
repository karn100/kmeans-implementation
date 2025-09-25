import numpy as np

class Kmeans_scratch:
    def __init__(self,k = 3,max_iters = 100,n_init = 10,tol = 1e-4,init = "kmeans++",distance = "euclidean",random_state = None,verbose = False):

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

    def _init_centroid(self,X,):
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        if self.init == "random":
        
            indcs = rng.choice(n_samples,self.k,replace=False)
            return X[indcs]

        elif self.init == "kmeans++":
            centroids = []
            centroids.append(X[rng.integers(n_samples)])

            for _ in range(1,self.k):
                d = np.min(self._compute_distance(X,np.array(centroids)),axis=1)
                probs = d**2/np.sum(d**2)
                next_centroid = X[rng.choice(n_samples,p=probs)] 
                centroids.append(next_centroid)
            return np.array(centroids)
        
        else:
            raise ValueError("init must be random or kmeans++")
    
    def _compute_distance(self,X,centroids):

        if self.distance == "euclidean":
            return np.linalg.norm(X[:,np.newaxis] - centroids,axis=2)
        
        elif self.distance == "manhattan":
            return np.linalg.norm