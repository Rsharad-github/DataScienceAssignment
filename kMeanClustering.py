# Step 1: Libraries import karte hain
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Step 2: Synthetic dataset create karte hain
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Step 3: Data ko visualize karte hain
plt.scatter(X[:, 0], X[:, 1], s=50, color='gray')
plt.title("Original Dataset (Unlabeled)")
plt.show()

# Step 4: k-Means model train karte hain
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Step 5: Predicted clusters plot karte hain
y_kmeans = kmeans.predict(X)

# Cluster centroids aur data points plot
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("k-Means Clustering Result")
plt.show()
