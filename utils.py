import math
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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


def plot_all_agents(all_agent_predictions, coin, all_agent_actuals=None):
    """Plot predictions and actuals for all agents in a single GUI window."""
    num_agents = len(all_agent_predictions)
    rows = math.ceil(math.sqrt(num_agents))
    cols = math.ceil(num_agents / rows)
    plt.figure(figsize=(15, rows * 5))

    for i in range(num_agents):
        predictions = all_agent_predictions[i]
        plt.subplot(rows, cols, i + 1)
        plt.plot(predictions, label=f"Agent {i+1} Predictions")
        if all_agent_actuals:
            actuals = all_agent_actuals[i]
            plt.plot(actuals, label=f"Agent {i+1} Actuals", alpha=0.7)
        plt.title(f"{coin} Price Prediction by Agent {i+1}")
        plt.xlabel("Days")
        plt.ylabel("Price in $")
        # plt.yticks([])
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
