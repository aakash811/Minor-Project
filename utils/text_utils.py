import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def clean_text(text):
    # Remove headers, multiple newlines, referencesa
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'TABLE \d+.*?', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    return text.strip()

def clean_and_split_sentences(text):
    cleaned = clean_text(text)
    return sent_tokenize(cleaned)

# from sentence_transformers import SentenceTransformer
# from typing import List
# import numpy as np

# model = SentenceTransformer('all-MiniLM-L6-v2') 

# def embed_sentences(sentences: List[str]) -> np.ndarray:
#     """Returns embeddings for a list of sentences using SBERT."""
#     embeddings = model.encode(sentences, show_progress_bar=True)
#     return embeddings


# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np

# def filter_relevant_sentences(sentences, keywords=None, top_k=0.5):
#     """
#     Filter sentences based on heuristic scoring combining:
#     - TF-IDF sentence score
#     - Sentence position score
#     - Presence of keywords
    
#     Args:
#         sentences (list of str): List of sentences extracted from PDF.
#         keywords (list of str): Optional list of keywords to boost sentence relevance.
#         top_k (float): Fraction of sentences to keep (e.g., 0.5 means keep top 50%).

#     Returns:
#         filtered_sentences (list of str): Sentences deemed relevant.
#     """
#     n = len(sentences)
#     if n == 0:
#         return []

#     # 1. TF-IDF score per sentence: sum of TF-IDF weights of words in that sentence
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(sentences)  # shape: (n_sentences, n_features)
#     sentence_tfidf_scores = tfidf_matrix.sum(axis=1).A1  # sum over words for each sentence

#     # Normalize TF-IDF scores to 0-1
#     sentence_tfidf_scores = (sentence_tfidf_scores - sentence_tfidf_scores.min()) / (sentence_tfidf_scores.ptp() + 1e-8)

#     # 2. Sentence position score: earlier sentences get higher score, linearly decaying
#     position_scores = np.linspace(1.0, 0.0, n)

#     # 3. Keyword boost: if keywords present in sentence, boost score by 0.1
#     keyword_scores = np.zeros(n)
#     if keywords:
#         keywords_set = set([kw.lower() for kw in keywords])
#         for i, sent in enumerate(sentences):
#             words = set(sent.lower().split())
#             if keywords_set.intersection(words):
#                 keyword_scores[i] = 0.1

#     # Combine scores (weights can be tuned)
#     combined_scores = 0.5 * sentence_tfidf_scores + 0.4 * position_scores + 0.1 * keyword_scores

#     # Select top sentences by combined score
#     keep_count = max(1, int(top_k * n))
#     top_indices = combined_scores.argsort()[::-1][:keep_count]

#     # Return filtered sentences in original order
#     filtered_sentences = [sentences[i] for i in sorted(top_indices)]

#     return filtered_sentences

