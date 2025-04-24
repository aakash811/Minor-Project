import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')

def tokenize_sentences(text):
    return nltk.sent_tokenize(text)

def vectorize_sentences(sentences):
    if not sentences:
        raise ValueError("No valid sentences to vectorize.")
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(sentences).toarray()
    return vectors
