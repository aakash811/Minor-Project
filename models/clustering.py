from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
import numpy as np

def cluster_sentences(embeddings, method="kmeans", n_clusters=5):
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "spectral":
        model = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=42)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
    elif method == 'birch':
        model = Birch(n_clusters=n_clusters)
    else:
        raise ValueError(f"Invalid method: {method}. Choose from 'kmeans', 'spectral', 'agglomerative', or 'dbscan'.")
    
    labels = model.fit_predict(embeddings)
    return labels

# def get_cluster_representatives(sentences, embeddings, labels):
#     summary_sentences = []
#     for cluster_id in set(labels):
#         # Skip noise points labeled as -1 in DBSCAN
#         if cluster_id == -1:
#             continue
#         cluster_idxs = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
#         cluster_embs = embeddings[cluster_idxs]
#         centroid = cluster_embs.mean(axis=0)
#         distances = np.linalg.norm(cluster_embs - centroid, axis=1)
#         best_idx = cluster_idxs[np.argmin(distances)]
#         summary_sentences.append(sentences[best_idx])
#     return summary_sentences

# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
# from scipy.stats import mode
# import numpy as np

# def all_cluster_methods(embeddings, n_clusters=10):
#     results = []
    
#     results.append(KMeans(n_clusters=n_clusters).fit_predict(embeddings))
#     results.append(DBSCAN(eps=0.5, min_samples=5).fit_predict(embeddings))
#     results.append(AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings))
#     results.append(SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors').fit_predict(embeddings))
#     results.append(Birch(n_clusters=n_clusters).fit_predict(embeddings))
    
#     # Convert DBSCAN -1 noise to a new cluster to avoid issues
#     for i, labels in enumerate(results):
#         noise_indices = np.where(labels == -1)[0]
#         if len(noise_indices) > 0:
#             max_label = max(labels) + 1
#             results[i][noise_indices] = max_label

#     combined = np.array(results).T  # shape: (n_samples, n_methods)
#     consensus = mode(combined, axis=1).mode.flatten()
#     return consensus
