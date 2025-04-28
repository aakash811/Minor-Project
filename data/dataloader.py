import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(vocab_size=10000, max_len=100, num_samples=500):
    # Generate random integer sequences
    X = np.random.randint(1, vocab_size, size=(num_samples, max_len))
    y = np.random.randint(0, 2, size=(num_samples,))
    
    X = pad_sequences(X, maxlen=max_len)
    return X, y
