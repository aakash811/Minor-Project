# models/dl_scorer.py

def build_model_from_genome(vocab_size, input_length):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Add this alias function
def build_scoring_model(vocab_size, input_length):
    return build_model_from_genome(vocab_size, input_length)
