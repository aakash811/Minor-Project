from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
import numpy as np

def cluster_sentences(embeddings, method="kmeans", n_clusters=5):
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(embeddings)

    elif method == "spectral":
        model = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=42)
        labels = model.fit_predict(embeddings)

    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(embeddings)

    elif method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(embeddings)
    else:
        raise ValueError(f"Invalid method: {method}. Choose from 'kmeans', 'spectral', 'agglomerative', or 'dbscan'.")
    return labels

def get_cluster_representatives(sentences, embeddings, labels):
    summary_sentences = []
    for cluster_id in set(labels):
        # Skip noise points labeled as -1 in DBSCAN
        if cluster_id == -1:
            continue
        cluster_idxs = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        cluster_embs = embeddings[cluster_idxs]
        centroid = cluster_embs.mean(axis=0)
        distances = np.linalg.norm(cluster_embs - centroid, axis=1)
        best_idx = cluster_idxs[np.argmin(distances)]
        summary_sentences.append(sentences[best_idx])
    return summary_sentences
