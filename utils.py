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
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
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


def train(X, X_train, y_train, index, args, predict):
    tqdm_verbose = 1 if args.debug > 1 else 0
    callbacks = [TqdmCallback(verbose=tqdm_verbose)] if args.debug > 0 else []
    model = build_and_compile_model(X.shape[2], args.coin)
    if args.retrain:
        fold_info = f"by Fold {index+1:02d}/{args.folds:02d}"
        if args.debug > 0 and predict:
            cprint(f"Predict Future {fold_info}", "purple")
        elif args.debug > 0 and not predict:
            cprint(
                f"Calculate History {fold_info}", "yellow")
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
    argparser.add_argument("--debug", type=int, default=0)
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


def load_history(args, kf, X, y, scaler, data, train_pred, actuals):
    predictions = []
    actuals = []
    cprint("Load History", "yellow")
    for index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test, y_train, y_test = split(X, y, train_index, test_index)
        model = train(X, X_train, y_train, index, args, False)
        prediction = scaler.inverse_transform(model.predict(X_test, verbose=0))
        predictions.extend(prediction)
        actuals.extend(scaler.inverse_transform(y_test.reshape(-1, 1)))
    tst = data.index[test_index].to_pydatetime()
    train_pred = pd.DataFrame(prediction, index=tst, columns=["Prediction"])
    x = scaler.inverse_transform(y_test.reshape(-1, 1))
    actuals = pd.DataFrame(x, index=tst, columns=["Actual"])
    return model, train_pred, actuals


def predict_future(args, kf, X, y, scaler, data, real_pred, prediction_hours):
    real_preds = []
    cprint("Predict Future", "yellow")
    for index, (train_index, _) in enumerate(kf.split(X)):
        X_train, y_train = split(X, y, train_index)
        model = train(X, X_train, y_train, index, args, True)
        prediction = scaler.inverse_transform(
            model.predict(X[-prediction_hours:], verbose=0))
        real_preds.extend(prediction)
    last_day = data.index[-1]
    next_day = last_day + pd.Timedelta(hours=1)
    rpl = len(real_preds)
    future_dates = pd.date_range(start=next_day, periods=rpl, freq="H")
    res = pd.DataFrame(real_preds, index=future_dates, columns=["Prediction"])
    return res


def get_data(args):
    data = load_and_preprocess_data(args.coin)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)
    kf = KFold(n_splits=args.folds, shuffle=False)
    return data, scaler, X, y, kf


def get_weight_file_path(coin):
    path = "weights/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path + f"{coin}.h5"


# def performance_output(args, real_pred, mae, coin):
#     duration = args.prediction * 12
#     real_pred_concat = pd.concat(real_pred)
#     first_entry = real_pred_concat.tail(duration).iloc[0]
#     last_entry = real_pred_concat.tail(duration).iloc[-1]
#     first_entry_value = first_entry["Prediction"]
#     last_entry_value = last_entry["Prediction"]
#     diff = (last_entry_value - first_entry_value)
#     p_change = round((diff / first_entry_value) * 100, 2)
#
#     trend = f"{coin} is rising by" if p_change > 0 else f"{coin} is falling by"
#     color = "green" if p_change > 0 else "red"
#
#     cprint(f"{trend} {p_change}% within {duration/24} days.", color)
#     cprint(f" > First Prediction {first_entry_value}", color)
#     cprint(f" > Last Prediction  {last_entry_value}", color)
#     cprint(f" > MAE              {mae}", "green" if mae < 1 else "red")
