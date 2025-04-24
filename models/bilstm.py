from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Bidirectional

def build_bilstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64), input_shape=(input_shape, 1)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

