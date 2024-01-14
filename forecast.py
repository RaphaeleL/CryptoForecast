import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.regularizers import l2
from keras.optimizers.legacy import Adam
from keras.layers import Dense, LSTM, Conv1D, Flatten, Bidirectional, Dropout
from tqdm.keras import TqdmCallback
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import cprint, plot, plot_backtest, get_colored_text, extract_min_max
from validation import space


class CryptoForecast:
    def __init__(self, ticker=None, minutely=False):
        self.start_time = time.time()
        self.duration = time.time()
        self.args = self.parse_args()
        self.end_date = self.get_end_date()
        self.ticker = self.args.coin if ticker is None else ticker
        self.prediction_days = self.args.prediction
        self.should_retrain = self.args.retrain
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.stretch_factor = 24
        self.strechted = not minutely
        self.data = self.get_data(stretch=not minutely)
        self.X, self.y = self.create_x_y_split()
        self.weight_path = self.create_weight_path()
        self.model = self.create_nn()
        self.forecast_data = None

    def set_retrain(self, should_retrain):
        self.should_retrain = should_retrain

    def get_data(self, stretch=False, period="max", interval="1d"):
        stretch_factor = self.stretch_factor
        if not self.args.minutely:
            raw_data = yf.download(self.ticker, period=period, interval=interval, progress=False)
            data = raw_data[["Close"]]
            data.reset_index(inplace=True)
            data.set_index("Date", inplace=True)
            if stretch:
                data.index = pd.to_datetime(data.index)
                interval = f"{int(24 / stretch_factor)}H"
                stretched = data.resample(interval).interpolate(method="time")
                return stretched
            return data
        raw_data = yf.download(self.ticker, period=period, interval="1m", progress=False)
        data = raw_data[["Close"]]
        data.reset_index(inplace=True)
        data.set_index("Datetime", inplace=True)
        if self.args.debug:
            data = data[data.index <= self.end_date]
        return data

    def create_x_y_split(self):
        data = self.scaler.fit_transform(self.data)
        dataX, dataY = [], []
        for i in range(len(data) - 1):
            a = data[i:i+1, 0]
            dataX.append(a)
            dataY.append(data[i + 1, 0])
        return np.array(dataX).reshape(-1, 1, 1), np.array(dataY)

    def split(self, X, y, train_index, test_index):
        X_train = X[train_index]
        y_train = y[train_index]
        if test_index is not None:
            X_test = X[test_index]
            y_test = y[test_index]
            return X_train, X_test, y_train, y_test
        return X_train, y_train

    def create_nn(self):
        model = Sequential(
            [
                Conv1D(64, 1, activation="relu", input_shape=(1, self.X.shape[2])),
                Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
                Dropout(0.2),
                Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
                Dropout(0.2),
                Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
                Dropout(0.2),
                Flatten(),
                Dense(50, activation="relu", kernel_regularizer=l2(0.001)),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(0.001), loss="mse")
        if os.path.isfile(self.weight_path):
            model.load_weights(self.weight_path)
        return model

    def create_weight_path(self):
        path = "weights/"
        if not os.path.exists(path):
            os.makedirs(path)
        return path + f"{self.ticker}.h5"

    def train(self, X_train, y_train):
        callbacks = [TqdmCallback(verbose=1)] if self.should_retrain else []
        if self.should_retrain:
            self.model.fit(
                X_train,
                y_train,
                batch_size=self.args.batch_size,
                epochs=self.args.epochs,
                verbose=0,
                callbacks=callbacks,
            )
        else:
            if os.path.isfile(self.weight_path):
                self.model.load_weights(self.weight_path)

    def parse_args(self):
        argparser = argparse.ArgumentParser(description="Cryptocurrency Forecast")
        argparser.add_argument("--coin", type=str, default="LTC-EUR")
        argparser.add_argument("--batch_size", type=int, default=1024)
        argparser.add_argument("--epochs", type=int, default=200)
        argparser.add_argument("--folds", type=int, default=6)
        argparser.add_argument("--prediction", type=int, default=7)
        argparser.add_argument("--retrain", action="store_true")
        argparser.add_argument("--minutely", action="store_true")
        argparser.add_argument("--debug", action="store_true")
        argparser.add_argument("--auto", action="store_true")
        args = argparser.parse_args()
        return args

    def get_end_date(self):
        today = datetime.date(
                datetime.datetime.now().year,
                datetime.datetime.now().month,
                datetime.datetime.now().day - 7
            )
        return today

    def load_history(self, agent=-1, should_save=True):
        all_train_pred = pd.DataFrame()
        all_actuals_df = pd.DataFrame()

        kf = KFold(n_splits=self.args.folds, shuffle=False)

        def train_fold(train_index, test_index):
            X_train, X_test, y_train, y_test = self.split(self.X, self.y, train_index, test_index)
            self.train(X_train, y_train)
            pred = self.scaler.inverse_transform(self.model.predict(X_test, verbose=0))

            idx = self.data.index[test_index].to_pydatetime()
            train_pred_fold = pd.DataFrame(pred, index=idx, columns=["Prediction"])
            actuals = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            actuals_fold = pd.DataFrame(actuals, index=idx, columns=["Actual"])
            return train_pred_fold, actuals_fold

        with ThreadPoolExecutor(max_workers=self.args.folds) as executor:
            futures = [executor.submit(train_fold, train_index, test_index) for train_index, test_index in kf.split(self.X)]
            for future in as_completed(futures):
                train_pred_fold, actuals_fold = future.result()
                all_train_pred = pd.concat([all_train_pred, train_pred_fold])
                all_actuals_df = pd.concat([all_actuals_df, actuals_fold])

        if self.should_retrain and should_save and agent >= 0:
            self.model.save_weights(self.weight_path)
            cprint(f"Saved model weights to '{self.weight_path}'", "green")

        return all_train_pred, all_actuals_df

    def predict_future(self):
        def prepare_future_input():
            input_window_size = self.args.prediction * 24
            if isinstance(self.X, pd.DataFrame):
                self.X = self.X.values
            if len(self.X) < input_window_size:
                raise ValueError("Not enough data to prepare future input.")
            future_input = self.X[-input_window_size:]
            return future_input

        pred_h = self.prediction_days * 24

        self.train(self.X, self.y)
        future_input = prepare_future_input()
        future_prediction = self.model.predict(future_input, verbose=0)
        prediction = self.scaler.inverse_transform(future_prediction)
        next_day = self.data.index[-1] + pd.Timedelta(hours=1)
        future_dates = pd.date_range(start=next_day, periods=pred_h, freq="H")
        res = pd.DataFrame(prediction, index=future_dates, columns=["Prediction"])
        self.forecast_data = res

    def visualize(self):
        plot(self.forecast_data, self.ticker)

    def visualize_backtest(self, actual_data):
        plot_backtest(self.forecast_data, actual_data, self.ticker)

    def stop_time(self):
        self.duration = round((time.time() - self.start_time), 1)

    def show_result(self):
        ticker = get_colored_text(yf.Ticker(self.ticker).info["name"] + " (" + self.ticker.split('-')[1] + ")", "yellow")
        min_index, min_value, max_index, max_value, global_min_index, global_min_value, global_max_index, global_max_value = extract_min_max(self)
        change = (((max_value - min_value) / min_value) - 0.05) * 100

        min_str = f"{min_value:.2f} {self.ticker.split('-')[1]}"
        max_str = f"{max_value:.2f} {self.ticker.split('-')[1]}"

        global_min_str = f"{global_min_value:.2f} {self.ticker.split('-')[1]}"
        global_max_str = f"{global_max_value:.2f} {self.ticker.split('-')[1]}"

        print(ticker)
        min_m, max_m = "├── Minimum (Logic)", "├── Maximum (Logic)"
        rise_m, dur_m = "├── Trend", "├── Duration (Prediction)"
        gmin_m, gmax_m = "├── Minimum (Global)", "├── Maximum (Global)"
        print(f"{min_m} {space(min_m, 4)}", get_colored_text(f"{min_index}", "yellow"), "with", get_colored_text(f"{min_str}", "green"))
        print(f"{gmin_m} {space(gmin_m, 4)}", get_colored_text(f"{global_min_index}", "yellow"), "with", get_colored_text(f"{global_min_str}", "green"))
        print(f"{max_m} {space(max_m, 4)}", get_colored_text(f"{max_index}", "yellow"), "with", get_colored_text(f"{max_str}", "red"))
        print(f"{gmax_m} {space(gmax_m, 4)}", get_colored_text(f"{global_max_index}", "yellow"), "with", get_colored_text(f"{global_max_str}", "red"))
        print(f"{rise_m} {space(rise_m, 4)}", get_colored_text(f"{format(change, '.2f') + '%'}", "green" if change > 0 else "red"))
        print(f"{dur_m} {space(dur_m, 4)}", f"{self.duration}s")

    def backtest(self):
        self.load_history()
        self.predict_future()

        backtest_period = yf.download(
                self.ticker,
                start=self.end_date,
                end=datetime.datetime.now(),
                interval="1d",
                progress=False
        )
        actual_data = backtest_period[["Close"]]
        return actual_data
