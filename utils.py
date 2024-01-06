import os
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Bidirectional, Dropout
from keras.regularizers import l2
from keras.optimizers.legacy import Adam
from tqdm.keras import TqdmCallback


def load_and_preprocess_data(ticker):
    data = yf.download(ticker)
    data = data[["Close"]]
    data.reset_index(inplace=True)
    data.set_index("Date", inplace=True)
    return stretch_data(data)


def stretch_data(data, stretch_factor=24):
    data.index = pd.to_datetime(data.index)
    interval = f"{int(24 / stretch_factor)}H"
    stretched = data.resample(interval).interpolate(method="time")
    return stretched


def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)


def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - 1):
        a = dataset[i: i + 1, 0]
        dataX.append(a)
        dataY.append(dataset[i + 1, 0])
    return np.array(dataX).reshape(-1, 1, 1), np.array(dataY)


def plot(coin, val, prediction):
    pre_days = prediction * 12
    formatter = mdates.DateFormatter("%Y-%m-%d - %H:%M")
    x = val.head(pre_days).index
    y = val["Prediction"][:pre_days]
    plt.plot(x, y, label="Prediction", alpha=0.7)
    plt.title(f"{coin} Future Predictions")
    plt.xlabel("Days")
    plt.ylabel(f"Price in {coin.split('-')[1]}")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
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


def build_and_compile_model(num_features, coin):
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
    if os.path.isfile(path):
        model.load_weights(path)
    return model


def train(args, X, X_train, y_train, predict):
    callbacks = [TqdmCallback(verbose=1)] if args.retrain else []
    model = build_and_compile_model(X.shape[2], args.coin)
    if args.retrain:
        model.fit(
            X_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=0,
            callbacks=callbacks,
        )
    else:
        path = get_weight_file_path(args.coin)
        if os.path.isfile(path):
            model.load_weights(path)
    return model


def argument_parser():
    argparser = argparse.ArgumentParser(
        description="Cryptocurrency Price Prediction")
    argparser.add_argument("--coin", type=str, default="LTC-EUR")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--epochs", type=int, default=5)
    argparser.add_argument("--folds", type=int, default=2)
    argparser.add_argument("--prediction", type=int, default=7)
    argparser.add_argument("--retrain", action="store_true")
    args = argparser.parse_args()
    return args


def cprint(to_print, color, end="\n"):
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


def load_history(args, X, y, scaler, data):
    all_train_pred = pd.DataFrame()
    all_actuals_df = pd.DataFrame()

    cprint("Load History", "yellow")

    kf = KFold(n_splits=args.folds, shuffle=False)
    for index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test, y_train, y_test = split(X, y, train_index, test_index)
        model = train(args, X, X_train, y_train, False)
        prediction = scaler.inverse_transform(model.predict(X_test, verbose=0))

        index = data.index[test_index].to_pydatetime()
        train_pred_fold = pd.DataFrame(
            prediction, index=index, columns=["Prediction"])
        all_train_pred = pd.concat([all_train_pred, train_pred_fold])

        actuals = scaler.inverse_transform(y_test.reshape(-1, 1))
        actuals_fold = pd.DataFrame(actuals, index=index, columns=["Actual"])
        all_actuals_df = pd.concat([all_actuals_df, actuals_fold])

    return model, all_train_pred, all_actuals_df


def predict_future(args, X, y, scaler, data, prediction_hours):
    cprint("Predict Future", "purple")

    model = train(args, X, X, y, True)
    future_input = prepare_future_input(X, prediction_hours)
    prediction = scaler.inverse_transform(
        model.predict(future_input, verbose=0))

    next_day = data.index[-1] + pd.Timedelta(hours=1)
    future_dates = pd.date_range(
        start=next_day, periods=prediction_hours, freq="H")

    res = pd.DataFrame(prediction, index=future_dates, columns=["Prediction"])
    return res


def prepare_future_input(X, prediction_hours, input_window_size=24*7):
    if isinstance(X, pd.DataFrame):
        X = X.values

    if len(X) < input_window_size:
        raise ValueError("Not enough data to prepare future input.")

    future_input = X[-input_window_size:]

    return future_input


def preprocess():
    args = argument_parser()
    data = load_and_preprocess_data(args.coin)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)
    return args, data, scaler, X, y


def postprocess(args, prd):
    plot(args.coin, prd, args.prediction)


def get_weight_file_path(coin):
    path = "weights/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path + f"{coin}.h5"
