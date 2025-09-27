from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scratch import Kmeans_scratch

X,_ = make_blobs(n_samples=500,centers=4,cluster_std=0.6,random_state=42)
model = Kmeans_scratch(k=4,n_init=5,init="kmeans++",random_state=42,verbose=True)
model.fit(X)

plt.scatter(X[:,0],X[:,1],c = model.labels,cmap='viridis',s = 30)
plt.scatter(model.centroids[:,0],model.centroids[:,1],c = 'red',s = 250,marker='*')
plt.title(f"Kmeans from Scratch with WCSS : {model.inertia:.2f}")
plt.show()