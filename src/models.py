from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM
from keras.layers import Flatten, Dropout, Dense, BatchNormalization
from keras.regularizers import l2
from keras.layers import Normalization


def bitcoin(cf):
    return Sequential([
        Normalization(input_shape=(1, cf.X.shape[2])),

        Conv1D(64, 3, activation="relu", padding="same"),
        MaxPooling1D(1),
        BatchNormalization(),
        Conv1D(128, 3, activation="relu", padding="same"),
        MaxPooling1D(1),
        BatchNormalization(),

        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        Dropout(0.3),

        Flatten(),

        Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(1)
    ])


def default_model(cf):
    return Sequential([
        Conv1D(64, 1, activation="relu", input_shape=(1, cf.X.shape[2])),

        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        Dropout(0.2),

        Flatten(),

        Dense(50, activation="relu", kernel_regularizer=l2(0.001)),
        Dense(1),
    ])
