import os
import math
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
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

def plot_all(real_pred, prediction):
    """Plot future Predictions"""
    pre_days = prediction * 12
    print("***", pre_days)
    for val in real_pred:
        if val.empty: continue
        plt.plot(val.index[:pre_days], val["Prediction"][:pre_days], label=f"Future Predictions")
    
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d - %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot(coin, agent, train, test, val, prediction):
    """Plot predictions and actuals for the best agents in a single GUI window."""
    length = 2
    _, axs = plt.subplots(length, 1, figsize=(15, 10))

    train, test, val = train[agent], test[agent], val[agent]
    
    axs[0].plot(train.index, train["Prediction"], label=f"Train Predictions")
    axs[0].plot(test.index, test["Actual"], label=f"Test Actuals", alpha=0.7)
    axs[0].set_title(f"{coin} Train/Test by Agent {agent+1}")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Price")
    axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=90))

    pre_days = prediction * 12 
    axs[1].plot(val.head(pre_days).index, val['Prediction'][:pre_days], label=f"Actual Prediction", alpha=0.7)
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

def build_and_compile_model(num_features, coin):
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
    path = get_weight_file_path(coin)
    if os.path.isfile(path): model.load_weights(path)
    return model

def train(X, X_train, y_train, batch_size, epochs, debug_level, args, agent, index, debug, predict=False):
    """Train the model."""
    tqdm_verbose = 1 if debug_level > 1 else 0
    callbacks = [TqdmCallback(verbose=tqdm_verbose)] if debug_level > 0 else []
    model = build_and_compile_model(X.shape[2], args.coin)
    if args.retrain:
        agent_info = f" for Agent {agent+1:02d}/{args.agents:02d}"
        fold_info = f"by Fold {index+1:02d}/{args.folds:02d}"
        if debug > 0 and predict: 
            print_colored(f"Predict Future {agent_info} {fold_info}", "purple")
        elif debug > 0 and not predict: 
            print_colored(f"Calculate History {agent_info} {fold_info}", "yellow")
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=callbacks,
        )
    else:
        path = get_weight_file_path(args.coin)
        if os.path.isfile(path): model.load_weights(path)
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
    argparser.add_argument("--plot", action="store_true")
    argparser.add_argument("--debug", type=int, default=2)
    argparser.add_argument("--retrain", action="store_true")
    argparser.add_argument("--save", action="store_true")
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

def load_history(args, kf, X, y, agent, scaler, data, test_pred, test_actu):
    predictions = []
    actuals = []
    for index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test, y_train, y_test = split(X, y, train_index, test_index)
        model = train(X, X_train, y_train, args.batch_size, args.epochs, args.debug, args, agent, index, args.debug)
        prediction = scaler.inverse_transform(model.predict(X_test, verbose=0))
        predictions.extend(prediction)
        actuals.extend(scaler.inverse_transform(y_test.reshape(-1, 1)))
    test_dates = data.index[test_index].to_pydatetime()
    test_pred.append(pd.DataFrame(prediction, index=test_dates, columns=['Prediction']))
    test_actu.append(pd.DataFrame(scaler.inverse_transform(y_test.reshape(-1, 1)), index=test_dates, columns=['Actual']))
    return model, test_pred, test_actu

def predict_future(args, kf, X, y, agent, scaler, data, real_pred, prediction_hours):
    real_predictions = []
    for index, (train_index, _) in enumerate(kf.split(X)):
        X_train, y_train = split(X, y, train_index) 
        model = train(X, X_train, y_train, args.batch_size, args.epochs, args.debug, args, agent, index, args.debug, True)
        prediction = scaler.inverse_transform(model.predict(X[-prediction_hours:], verbose=0))
        real_predictions.extend(prediction)
    last_day = data.index[-1]
    next_day = last_day + pd.Timedelta(hours=1)
    future_dates = pd.date_range(start=next_day, periods=len(real_predictions), freq='H')
    real_pred.append(pd.DataFrame(real_predictions, index=future_dates, columns=['Prediction']))
    return real_pred

def get_data(args):
    data = load_and_preprocess_data(args.coin)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)
    kf = KFold(n_splits=args.folds, shuffle=False)
    return data, scaler, X, y, kf

def get_weight_file_path(coin):
    path = f"weights/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path + f"{coin}.h5"

def performance_output(args, real_pred, best_agent, coin, test_pred, test_actu):
    duration = args.prediction * 12
    first_entry = real_pred[best_agent].tail(duration).iloc[0]
    last_entry = real_pred[best_agent].tail(duration).iloc[-1]
    first_entry_value = first_entry['Prediction']
    last_entry_value = last_entry['Prediction']
    percentage_change = round(((last_entry_value - first_entry_value) / first_entry_value) * 100, 2)
    trend = "rising" if percentage_change > 0 else "falling"
    color = "green" if percentage_change > 0 else "red"

    print_colored(f"{coin} is {trend} by {percentage_change}% within {duration/24} days.", color)
    if args.debug > 1:
        print_colored(f" > First Prediction {first_entry_value}", color)
        print_colored(f" > Last Prediction  {last_entry_value}", color)
    if args.plot or args.debug > 1:
        plot(args.coin, best_agent, test_pred, test_actu, real_pred, args.prediction)

def get_model_with_weights(X, args, agent):
    model = build_and_compile_model(X.shape[2], args.coin)
    weight_file = get_weight_file_path(args.coin)
    if os.path.isfile(weight_file):
        model.load_weights(weight_file)
        print_colored(f"Loading History for Agent {agent+1:02d}/{args.agents:02d}", "blue")
    else: 
        print_colored(f"ERROR: There is no Weight File for {args.coin} under {weight_file}", "red")
        print_colored(f"       Consider to run following Command, to create the missing File:", "red")
        print_colored(f"       $ python3 forecast.py --coin {args.coin}", "red")
    return model