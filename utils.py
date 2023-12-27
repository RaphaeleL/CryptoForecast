import math
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Bidirectional, Dropout
from keras.regularizers import l2
from keras.optimizers.legacy import Adam

def load_and_preprocess_data(ticker):
    """Load and preprocess data from the given path."""
    data = yf.download(ticker)
    data = data[["Close"]]
    data.reset_index(inplace=True)
    data.set_index("Date", inplace=True)
    return stretch_data(data)

def stretch_data(data, stretch_factor=24):
    """Stretch daily data to a specified hourly interval."""
    data.index = pd.to_datetime(data.index)
    interval = f"{int(24 / stretch_factor)}H"
    stretched = data.resample(interval).interpolate(method="time")
    return stretched

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

def plot_all(coin, best_agent, val):
    """Plot predictions and actuals for all agents in a single GUI window."""
    num_agents = len(val)
    rows = math.ceil(math.sqrt(num_agents))
    cols = math.ceil(num_agents / rows)
    plt.figure(figsize=(15, rows * 5))
    
    for i in range(num_agents):
        if i == best_agent: title = f"{coin} Price Prediction by BEST Agent {i+1}"
        else: title = f"{coin} Price Prediction by Agent {i+1}"
        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(val[i].index, val[i]["Prediction"], label=f"Agent {i+1} Real Predictions")
        ax.set_title(title)
        ax.set_xlabel("Days")
        ax.set_ylabel("Price in $")
        ax.legend()
        ax.grid(True)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=30)) 
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # plt.savefig(f"images/{coin}_{pd.to_datetime("today").strftime("%Y-%m-%d")}.png")
    plt.tight_layout()
    plt.show()

def plot(coin, agent, train, test, val, prediction):
    """Plot predictions and actuals for the best agents in a single GUI window."""
    length = 2
    _, axs = plt.subplots(length, 1, figsize=(15, 10))
    
    axs[0].plot(train.index, train["Prediction"], label=f"Train Predictions")
    axs[0].plot(test.index, test["Actual"], label=f"Test Actuals", alpha=0.7)
    axs[0].set_title(f"{coin} Train/Test by Agent {agent+1}")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Price")
    axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=90))

    pre_days = prediction * 12 
    axs[1].plot(test.tail(pre_days).index, test['Actual'][-pre_days:], label=f"Test Actuals", alpha=0.7)
    axs[1].set_title(f"{coin} Future Predictions by Agent {agent+1}")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Price")
    axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))

    for i in range(length):
        axs[i].legend()
        axs[i].grid(True)
        axs[i].xaxis_date()
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d - %H:%M"))
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def split(X, y, train_index, test_index=None):
    """Split the dataset into training and testing sets."""
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
            Bidirectional(LSTM(50, activation="relu", return_sequences=True)),
            Dropout(0.2),
            Flatten(),
            Dense(50, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def train(X, X_train, y_train, batch_size, epochs):
    """Train the model."""
    model = build_and_compile_model(X.shape[2])
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
    )
    return model

def argument_parser():
    """Parse command line arguments."""
    argparser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction")
    argparser.add_argument("--coin", type=str, default="LTC-EUR")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--epochs", type=int, default=5)
    argparser.add_argument("--agents", type=int, default=6)
    argparser.add_argument("--folds", type=int, default=2)
    argparser.add_argument("--prediction", type=int, default=7)
    argparser.add_argument("--show_all", action="store_true")
    args = argparser.parse_args()
    return args

def check(first_pred, last_actual, p=0.05):
    """Check if the prediction is within the given percentage of the actual value."""
    if first_pred >= last_actual * (1-p) and first_pred <= last_actual * (1+p):
        return True
    return False

def print_colored(to_print, color, end="\n"):
    """Print text in the given color."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "end": "\033[0m",
    }
    print(f"{colors.get(color, '')}{to_print}{colors['end']}", end=end)

def print_data(data, best_agent):
    """Print the dataset."""
    for agent in range(len(data)):
        color = "green" if agent == best_agent else "red"
        border = "*" * 31 
        end_border = "*" * 17 
        print_colored(border, color)
        print_colored(f"***** Agent {agent+1} {end_border}", color)
        print_colored(border, color)
        print(data[agent])

def select_best_agent(performance_data):
    """Select the best agent based on performance data."""
    best_agent_index = performance_data.index(min(performance_data))
    return best_agent_index

def need_retraining(best_agent, performance_data, threshold=0.1):
    """Check if the best agent"s performance is above a certain threshold, indicating a need for retraining."""
    return performance_data[best_agent] > threshold

def evaluate_agent_performance(test_actu, test_pred):
    """Evaluate the performance of each agent using Mean Absolute Error."""
    performance_data = []
    for actual, prediction in zip(test_actu, test_pred):
        mae = mean_absolute_error(actual["Actual"], prediction["Prediction"])
        performance_data.append(mae)
    return performance_data

def print_agent_performance_overview(agents, performance_data, best_agent):
    color = {x : "red" for x in range(agents)}
    for agent in range(agents):
        if need_retraining(agent, performance_data, 2.0):
            # TODO: Retrain
            best = "*"
        else: 
            best = " "
        if agent == best_agent:
            color[agent] = "green"
        print_colored(f"{best} Agent {agent+1:02d} Performance: {performance_data[agent]:05.2f}", color[agent])
