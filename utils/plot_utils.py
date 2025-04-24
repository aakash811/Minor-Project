from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_clusters(embeddings, labels):
    reduced = PCA(n_components=2).fit_transform(embeddings)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10')
    plt.title("Clustered Sentences")
    plt.colorbar()
    plt.show()
