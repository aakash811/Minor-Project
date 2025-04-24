from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
import numpy as np

def kmeans_cluster(embeddings, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(embeddings)
    return labels

def dbscan_cluster(embeddings, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(embeddings)
    return labels

def agglomerative_cluster(embeddings, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(embeddings)
    return labels

def spectral_cluster(embeddings, n_clusters=5):
    model = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=42)
    labels = model.fit_predict(embeddings)
    return labels

def birch_cluster(embeddings, n_clusters=5):
    model = Birch(n_clusters=n_clusters)
    labels = model.fit_predict(embeddings)
    return labels

def gaussian_mixture_cluster(embeddings, n_components=5):
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(embeddings)
    labels = model.predict(embeddings)
    return labels
