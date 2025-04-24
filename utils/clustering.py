from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

clustering_model = build_clustering(n_clusters=3)
clustering_model.fit(X)

labels = clustering_model.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("KMeans Clustering")
plt.show()
