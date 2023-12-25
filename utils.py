import math
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from keras import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Bidirectional, Dropout
from keras.regularizers import l2
from keras.optimizers.legacy import Adam
from tqdm.keras import TqdmCallback

def plot(coin, data, exit_after=True):
    """Plot the given data."""
    plt.title(f"{coin} Price History")
    plt.plot(data)
    plt.grid(True)
    plt.show()
    if exit_after:
        exit()


def load_and_preprocess_data(coin, reverse=False):
    """Load and preprocess data from the given path."""
    data = pd.DataFrame(yf.Ticker(coin).history(period="max"))
    data_columns = data.columns.to_list()
    data_columns.remove("Close")
    for column in data_columns:
        if column in data.columns:
            data = data.drop(column, axis=1)
    data = (
        data.replace("\.", "", regex=True).replace(",", ".", regex=True).astype(float)
    )
    if reverse:
        data = data.iloc[::-1].reset_index(drop=True)
    return data


def normalize_data(data):
    """Normalize data using Min-Max Scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)


def create_dataset(dataset):
    """Create dataset matrix for training and testing using the full dataset."""
    dataX, dataY = [], []
    for i in range(len(dataset) - 1):
        a = dataset[i : i + 1, 0]
        dataX.append(a)
        dataY.append(dataset[i + 1, 0])
    return np.array(dataX).reshape(-1, 1, 1), np.array(dataY)


def plot_predictions(testPredict, coin, testY=None):
    """Plot the actual vs predicted prices."""
    if testY is not None:
        plt.plot(testY[::-1], label="Actual Price")
    title = f"{coin} Price Prediction"
    plt.plot(testPredict[::-1], label="Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Price in $")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_agents(coin, train, test, val):
    """Plot predictions and actuals for all agents in a single GUI window."""
    num_agents = len(train)
    rows = math.ceil(math.sqrt(num_agents))
    cols = math.ceil(num_agents / rows)
    plt.figure(figsize=(15, rows * 5))
    for i in range(num_agents):
        plt.subplot(rows, cols, i + 1)
        plt.plot(train[i], label=f"Agent {i+1} Predictions")
        plt.plot(test[i], label=f"Agent {i+1} Actuals", alpha=0.7)
        plt.plot(val[i], label=f"Agent {i+1} Real Predictions")
        plt.title(f"{coin} Price Prediction by Agent {i+1}")
        plt.xlabel("Days")
        plt.ylabel("Price in $")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def split(X, y, train_index, test_index=None):
    X_train = X[train_index]
    y_train = y[train_index]
    if test_index is not None:
        X_test = X[test_index]
        y_test = y[test_index]
        return X_train, X_test, y_train, y_test
    return X_train, y_train

def build_and_compile_model(num_features):
    """Build and compile the Keras Sequential model."""
    model = Sequential(
        [
            Conv1D(64, 1, activation="relu", input_shape=(1, num_features)),
            Bidirectional(LSTM(50, activation="relu", return_sequences=True)),
            Bidirectional(LSTM(50, activation="relu", return_sequences=True)),
            Dropout(0.2),
            Flatten(),
            Dense(50, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def train(X, y, X_train, y_train, batch_size, epochs):
    model = build_and_compile_model(X.shape[2])
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        callbacks=[TqdmCallback(verbose=0)],
    )
    return model

def argument_parser():
    argparser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction")
    argparser.add_argument("--coin", type=str, default="eth")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--agents", type=int, default=1)
    argparser.add_argument("--folds", type=int, default=5)
    argparser.add_argument("--plot_coin", action="store_true")
    argparser.add_argument("--reverse", action="store_true")
    args = argparser.parse_args()
    return args

def check(first_pred, last_actual, p=0.10):
    if first_pred >= last_actual * (1-p) and first_pred <= last_actual * (1+p):
        return True
    return False