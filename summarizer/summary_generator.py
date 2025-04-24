# import numpy as np

# def select_representative_sentences(sentences, embeddings, cluster_labels, top_n=1):
#     """
#     For each cluster, select the top_n sentences closest to the cluster centroid.
    
#     Args:
#         sentences (list of str): All candidate sentences.
#         embeddings (np.array): Sentence embeddings (shape: num_sentences x embedding_dim).
#         cluster_labels (np.array): Cluster labels for each sentence.
#         top_n (int): Number of representative sentences to select per cluster.
        
#     Returns:
#         summary_sentences (list of str): Selected sentences for the summary.
#     """
#     unique_clusters = set(cluster_labels)
#     summary_sentences = []

#     for cluster in unique_clusters:
#         if cluster == -1:
#             continue  # skip noise
#         idxs = np.where(cluster_labels == cluster)[0]
#         cluster_embs = embeddings[idxs]
#         centroid = np.mean(cluster_embs, axis=0)

#         # Compute distance of each sentence in cluster to centroid
#         distances = np.linalg.norm(cluster_embs - centroid, axis=1)
#         sorted_idxs = idxs[np.argsort(distances)]

#         # Select top_n closest sentences
#         for idx in sorted_idxs[:top_n]:
#             summary_sentences.append(sentences[idx])

#     return summary_sentences
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

def select_representative_sentences(sentences, embeddings, cluster_labels, top_n=1):
    """
    For each cluster, select top_n sentences closest to the cluster centroid as summary.
    """
    summary_sentences = []
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        if label == -1:
            continue  # skip noise if any
        indices = np.where(cluster_labels == label)[0]
        cluster_embeddings = embeddings[indices]
        centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
        closest, _ = pairwise_distances_argmin_min(centroid, cluster_embeddings)
        
        # Pick top_n sentences (closest ones)
        for i in closest[:top_n]:
            summary_sentences.append(sentences[indices[i]])
    return summary_sentences
