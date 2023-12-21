import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(coin):
    """ Load and preprocess data from the given path. """
    data = pd.DataFrame(yf.Ticker(coin).history(period="max"))
    data_columns = data.columns.to_list()
    data_columns.remove("Close")
    for column in data_columns:
        if column in data.columns:
            data = data.drop(column, axis=1)
    data = data.replace("\.", "", regex=True).replace(",", ".", regex=True).astype(float)
    data = data.iloc[::-1].reset_index(drop=True)
    return data

def normalize_data(data):
    """ Normalize data using Min-Max Scaler. """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)

def create_dataset(dataset):
    """ Create dataset matrix for training and testing using the full dataset. """
    dataX, dataY = [], []
    for i in range(len(dataset) - 1):
        a = dataset[i:i + 1, 0]
        dataX.append(a)
        dataY.append(dataset[i + 1, 0])
    return np.array(dataX).reshape(-1, 1, 1), np.array(dataY)

def plot_predictions(testPredict, coin="n/a", testY=None):
    """ Plot the actual vs predicted prices. """
    if testY is not None: plt.plot(testY[::-1], label="Actual Price")
    title = f"{coin} Price Prediction"
    plt.plot(testPredict[::-1], label="Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Price in $")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()