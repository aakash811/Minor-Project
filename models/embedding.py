import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Train Word2Vec model (once, or load a pre-trained one)
def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    tokenized = [word_tokenize(sent.lower()) for sent in sentences]
    model = Word2Vec(sentences=tokenized, vector_size=vector_size, window=window, min_count=min_count)
    return model

# Get sentence embedding by averaging word vectors
def get_embeddings(sentences, word2vec_model):
    embeddings = []
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        vecs = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
        if vecs:
            avg_vec = np.mean(vecs, axis=0)
        else:
            avg_vec = np.zeros(word2vec_model.vector_size)
        embeddings.append(avg_vec)
    return np.array(embeddings)
