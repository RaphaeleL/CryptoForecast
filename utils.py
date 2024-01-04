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
from tqdm.keras import TqdmCallback


def load_and_preprocess_data(ticker):
    """Load and preprocess data from the given path."""
    data = yf.download(ticker)
    data = data[["Close"]]
    data.reset_index(inplace=True)
    data.set_index("Date", inplace=True)
    return stretch_data(data)


def stretch_data(data, stretch_factor=4):
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
    """Create dataset matrix for training and testing using the full dataset"""
    dataX, dataY = [], []
    for i in range(len(dataset) - 1):
        a = dataset[i: i + 1, 0]
        dataX.append(a)
        dataY.append(dataset[i + 1, 0])
    return np.array(dataX).reshape(-1, 1, 1), np.array(dataY)


def plot(coin, agent, val, prediction, mae_score, p_trend):
    """Plot predictions for the best agent"""
    val_agent = val[agent]
    pre_days = prediction * 12
    pre_days = min(pre_days, len(val_agent))
    trend = "rising" if p_trend > 0 else "falling"
    plt.figure(figsize=(10, 5))
    indexes = val_agent.head(pre_days).index
    labels = val_agent['Prediction'][:pre_days]
    plt.plot(indexes, labels, label="Prediction", alpha=0.7)
    plt.title(f"{coin} Prediction by #{agent+1} with MAE {mae_score:.2f} -\
            It is {trend} by {p_trend}% within {pre_days/24} days.")
    plt.xlabel("Days")
    plt.ylabel("Price")
    formatter = mdates.DateFormatter("%Y-%m-%d - %H:%M")
    locator = mdates.DayLocator(interval=1)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
    plt.grid(True)
    plt.legend()
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
            Dropout(0.2),
            Flatten(),
            Dense(50, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model


def train_model(X, X_train, y_train, args):
    """Train the model."""
    batch_size, epochs, debug_level = args.batch_size, args.epochs, args.debug
    tqdm_verbose = 1 if debug_level > 1 else 0
    callbacks = [TqdmCallback(verbose=tqdm_verbose)] if debug_level > 0 else []
    model = build_and_compile_model(X.shape[2])
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
    )
    return model


def argument_parser():
    """Parse command line arguments."""
    argparser = argparse.ArgumentParser(
        description="Cryptocurrency Price Prediction")
    argparser.add_argument("--coin", type=str, default="LTC-EUR")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--agents", type=int, default=6)
    argparser.add_argument("--folds", type=int, default=6)
    argparser.add_argument("--prediction", type=int, default=7)
    argparser.add_argument("--debug", type=int, default=2)
    args = argparser.parse_args()
    return args


def cprint(to_print, color, end="\n"):
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


def select_best_agent(performance_data):
    """Select the best agent based on performance data."""
    best_agent_index = performance_data.index(min(performance_data))
    return best_agent_index


def evaluate_agent_performance(test, train):
    """Evaluate the performance of each agent using Mean Absolute Error."""
    performance_data = []
    for actual, prediction in zip(test, train):
        mae = mean_absolute_error(actual["Actual"], prediction["Prediction"])
        performance_data.append(mae)
    return performance_data


def load_history(args, kf, agent, X, y, scaler, data, train, test):
    """Load history for each agent."""
    predictions = []
    actuals = []
    cprint(
        f"Load History for Agent {agent+1:02d}/{args.agents:02d}", "yellow")
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = split(X, y, train_index, test_index)
        model = train_model(X, X_train, y_train, args)
        prediction = scaler.inverse_transform(model.predict(X_test, verbose=0))
        predictions.extend(prediction)
        actuals.extend(scaler.inverse_transform(y_test.reshape(-1, 1)))

    test_dates = data.index[test_index].to_pydatetime()
    train.append(pd.DataFrame(
        prediction, index=test_dates, columns=['Prediction']))
    test.append(pd.DataFrame(scaler.inverse_transform(
        y_test.reshape(-1, 1)), index=test_dates, columns=['Actual']))

    return train, test


def predict_future(args, kf, agent, X, y, scaler, data, val):
    """Predict future cryptocurrency prices."""
    valictions = []
    agent_str = f"{agent+1:02d}/{args.agents:02d}"
    cprint(f"Predict Future for Agent {agent_str}", "purple")
    for train_index, _ in kf.split(X):
        X_train, y_train = split(X, y, train_index)
        model = train_model(X, X_train, y_train, args)
        prediction = scaler.inverse_transform(model.predict(X, verbose=0))
        valictions.extend(prediction)

    last_day = data.index[-1]
    next_day = last_day + pd.Timedelta(hours=1)
    future_dates = pd.date_range(
        start=next_day, periods=len(valictions), freq='H')
    df = pd.DataFrame(valictions, index=future_dates, columns=['Prediction'])
    val.append(df)

    return val


def calculate_trend(prediction_duration, val, best_agent, coin):
    """Output the performance of the best agent."""
    first = val[best_agent].tail(prediction_duration).iloc[0]['Prediction']
    last = val[best_agent].tail(prediction_duration).iloc[-1]['Prediction']
    percentage = ((last - first) / first)
    return round(percentage * 100, 2)
