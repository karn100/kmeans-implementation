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

        if self.distance == "euclidean":               #L2 regularisation
            return np.linalg.norm(X[:,np.newaxis] - centroids,axis=2)
        
        elif self.distance == "manhattan":            #L1 regularisation
            return np.sum(np.abs(X[:,np.newaxis] - centroids),axis=2)
        else:
            raise ValueError("distance must be euclidean or manhattan")
    
    def _compute_inertia(self,X,labels,centroids):
        return np.sum(np.linalg.norm(X - centroids[labels],axis=1)**2)
    
    def fit(self,X):
        best_inertia = np.inf
        best_centroids,best_labels = None,None
        rng = np.random.default_rng(self.random_state)

        for run in range(self.n_init):
            centroids = self._init_centroid
            ineria_history = []
            for it in range(self.max_iters):
                distance = self._compute_distance(X,centroids)
                labels = np.argmin(distance,axis=1)

                new_centroids = np.array([
                    X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range
                ])
                inertia = self._compute_inertia(X,labels,new_centroids)
                ineria_history.append(inertia)
                if self.verbose:
                    print(f"Run {run+1}, Iteration {it+1}, WCSS: {inertia:.4f}")

                shift = np.linalg.norm(centroids - new_centroids,axis = 1).max()
                centroids = new_centroids
                if shift < self.tol:
                    break

                if inertia < best_inertia:
                    best_inertia = inertia
                    best_centroids = centroids
                    best_labels = labels
                    best_inertia_history = ineria_history
            
        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = best_inertia
        self.inertia_history = best_inertia_history
        return self
    
    def predict(self,X):
        distcance = self._compute_distance(X,self.centroids)
        return np.argmin(distcance,axis=1)
    
        

