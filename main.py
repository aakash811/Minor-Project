# import sys
# import numpy as np
# from utils.preprocess import tokenize_sentences, vectorize_sentences
# from models.ann import build_ann
# from models.cnn import build_cnn
# from models.clustering import build_clustering
# from models.lstm import build_lstm
# from models.bilstm import build_bilstm

# def summarize_supervised(text, model_name="bilstm", num_sentences=5):
#     sentences = tokenize_sentences(text)
#     vectors = vectorize_sentences(sentences)
#     labels = np.random.randint(0, 2, len(sentences))  # Dummy labels for training

#     # Choose model
#     model_map = {
#         "ann": build_ann,
#         "cnn": build_cnn,
#         "lstm": build_lstm,
#         "bilstm": build_bilstm
#     }
#     model_fn = model_map.get(model_name.lower(), build_bilstm)
#     model = model_fn(vectors.shape[1])

#     model.fit(vectors, labels, epochs=3, verbose=0)
#     scores = model.predict(vectors).flatten()
#     top_indices = np.argsort(scores)[-num_sentences:]

#     summary = ' '.join([sentences[i] for i in sorted(top_indices)])
#     return summary

# def summarize_clustering(text, num_sentences, n_clusters=2):
#     sentences = tokenize_sentences(text)
#     vectors = vectorize_sentences(sentences)

#     model = build_clustering(n_clusters=n_clusters)
#     model.fit(vectors)
#     labels = model.labels_

#     largest_cluster = np.bincount(labels).argmax()
#     selected_indices = np.where(labels == largest_cluster)[0][:num_sentences]

#     summary = ' '.join([sentences[i] for i in sorted(selected_indices)])
#     return summary

# if __name__ == "__main__":
#     with open("data/sample.txt", "r") as f:
#         text = f.read()

#     method = sys.argv[1] if len(sys.argv) > 1 else "bilstm"
#     print(f"\nSelected method: {method}\n")

#     if method.lower() == "clustering":
#         result = summarize_clustering(text, num_sentences=7)
#     else:
#         result = summarize_supervised(text, model_name=method, num_sentences=3)

#     print("\n=== SUMMARY ===\n")
#     print(result)


from utils.pdf_utils import extract_text_from_pdf
from utils.text_utils import clean_and_split_sentences
from models.embedding import train_word2vec, get_embeddings
from models.clustering import cluster_sentences
from models.dl_scorer import build_scoring_model
from models.dl_scorer_ga import run_ga
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def summarize_pdf(
    pdf_path=None,
    custom_text=None,
    clustering_methods=['agglomerative', 'kmeans', 'spectral', 'dbscan', 'birch'],
    cluster_ratio=0.5,
    epochs=200,
    plot=False,
    ga_params=None
):
    if custom_text:
        text = custom_text
    else:
        text = extract_text_from_pdf(pdf_path)
    sentences = clean_and_split_sentences(text)
    filtered_sentences = [s for s in sentences if 5 < len(s.split()) < 40]

    # Ensure there is text to summarize
    if not filtered_sentences:
        return []

    n_sentences = len(filtered_sentences)
    n_clusters = max(1, int(n_sentences * cluster_ratio)) 

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(filtered_sentences)
    sequences = tokenizer.texts_to_sequences(filtered_sentences)
    padded_seqs = pad_sequences(sequences, padding='post')

    w2v_model = train_word2vec(filtered_sentences)
    embeddings = get_embeddings(filtered_sentences, w2v_model)
    labels = np.array([1 if len(s.split()) > 10 else 0 for s in filtered_sentences]) 

    X_train, X_val, y_train, y_val = train_test_split(
        padded_seqs, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vocab_size = len(tokenizer.word_index) + 1
    input_length = padded_seqs.shape[1]

    if ga_params is None:
        ga_params = {
            "lstm_units": 64,
            "dropout_rate": 0.5,
            "lr": 0.001,
            "optimizer_name": "adam"
        }

    model = build_scoring_model(
        vocab_size,
        input_length,
        lstm_units=ga_params['lstm_units'],
        dropout_rate=ga_params['dropout_rate'],
        lr=ga_params['lr'],
        optimizer_name=ga_params['optimizer_name']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=[early_stop]
    )
    y_pred_probs = model.predict(X_val)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    scores = model.predict(padded_seqs).flatten()
    combined_summary_sentences = set()
    for method in clustering_methods:
        cluster_labels = cluster_sentences(embeddings, method, n_clusters)

        for cluster_id in set(cluster_labels):
            cluster_idxs = [i for i, lbl in enumerate(cluster_labels) if lbl == cluster_id]
            if not cluster_idxs:
                continue
            cluster_scores = scores[cluster_idxs]
            best_idx = cluster_idxs[np.argmax(cluster_scores)]
            combined_summary_sentences.add(filtered_sentences[best_idx])

    # cluster_labels = cluster_sentences(embeddings, method=clustering_method, n_clusters=n_clusters)

    # summary_sentences = []
    # for cluster_id in set(cluster_labels):
    #     cluster_idxs = [i for i, lbl in enumerate(cluster_labels) if lbl == cluster_id]
    #     cluster_scores = scores[cluster_idxs]
    #     best_idx = cluster_idxs[np.argmax(cluster_scores)]
    #     summary_sentences.append(filtered_sentences[best_idx])

    if plot:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        reduced_embs = pca.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))
        first_method_labels = cluster_sentences(embeddings, clustering_methods[0], n_clusters)
        scatter = plt.scatter(reduced_embs[:, 0], reduced_embs[:, 1], c=first_method_labels, cmap='tab10')
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.title(f"Sentence Embeddings Cluster Plot ({clustering_methods[0]})")
        plt.show()

    return list(combined_summary_sentences)

if __name__ == "__main__":
    pdf_file = "data/GenSumm_A_Joint_Framework_for_Multi-Task_Tweet_Classification_and_Summarization_Using_Sentiment_Analysis_and_Generative_Modelling.pdf"
    
    try:
        best_hyperparams = run_ga()
        summary = summarize_pdf(
            pdf_file,
            clustering_method="agglomerative",
            n_clusters=60,
            plot=True,
            ga_params=best_hyperparams
        )
        print("\n📝 SUMMARY:\n")
        for i, s in enumerate(summary, 1):
            print(f"{i}. {s}\n")
    except Exception as e:
        print(f"❌ Error during summarization: {e}")



# import os
# from utils.pdf_utils import extract_sentences_from_pdf
# from utils.embedding_utils import get_sentence_embeddings
# from clustering.clustering_methods import (
#     kmeans_cluster,
#     dbscan_cluster,
#     agglomerative_cluster,
#     spectral_cluster,
#     birch_cluster,
#     gaussian_mixture_cluster,
# )
# from clustering.snorkel_combiner import combine_cluster_labels
# from summarizer.summary_generator import select_representative_sentences
# import numpy as np

# def main(pdf_path, num_clusters=5, top_sentences_per_cluster=1):
#     print(f"Extracting sentences from {pdf_path} ...")
#     sentences = extract_sentences_from_pdf(pdf_path)
#     print(f"Extracted {len(sentences)} sentences.")

#     print("Generating embeddings ...")
#     embeddings = get_sentence_embeddings(sentences)

#     print("Running clustering methods ...")
#     labels_kmeans = kmeans_cluster(embeddings, n_clusters=num_clusters)
#     labels_dbscan = dbscan_cluster(embeddings)
#     labels_agg = agglomerative_cluster(embeddings, n_clusters=num_clusters)
#     labels_spectral = spectral_cluster(embeddings, n_clusters=num_clusters)
#     labels_birch = birch_cluster(embeddings, n_clusters=num_clusters)
#     labels_gmm = gaussian_mixture_cluster(embeddings, n_components=num_clusters)

#     print("Combining clustering outputs ...")
#     combined_labels = combine_cluster_labels([
#         labels_kmeans,
#         labels_dbscan,
#         labels_agg,
#         labels_spectral,
#         labels_birch,
#         labels_gmm
#     ])

#     print("Selecting representative sentences ...")
#     summary = select_representative_sentences(sentences, embeddings, combined_labels, top_n=top_sentences_per_cluster)

#     print("\n=== Summary ===")
#     for sent in summary:
#         print("-", sent)

# if __name__ == "__main__":
#     pdf_file = os.path.join("data", "GenSumm_A_Joint_Framework_for_Multi-Task_Tweet_Classification_and_Summarization_Using_Sentiment_Analysis_and_Generative_Modelling.pdf")  # Replace with your file
#     main(pdf_file)

