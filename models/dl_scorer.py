from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

def build_scoring_model(vocab_size, input_length, lstm_units=64, dropout_rate=0.5, lr=0.001, optimizer_name='adam'):
    optimizers = {
        'adam': Adam(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr),
        'sgd': SGD(learning_rate=lr)
    }
    optimizer = optimizers.get(optimizer_name.lower(), Adam(learning_rate=lr))

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
