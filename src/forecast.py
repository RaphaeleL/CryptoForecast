import os
import glob
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

from src.utils import (
    plot,
    create_cloud_path,
    get_dafault_bw_path
)


class CryptoForecast:
    def __init__(self):
        self.metric = ""
        self.args = self.parse_args()
        self.ticker = self.args.coin
        self.weight_path = create_cloud_path(self.args.path, ticker=self.ticker, typeof="weights", filetype="h5")
        self.should_retrain = self.args.retrain
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.raw_data = None
        self.data = self.get_data()
        self.X, self.y = self.create_x_y_split()
        self.model = self.create_nn()
        self.forecast_data = None
        self.future_days = self.args.future

    def set_retrain(self, should_retrain):
        self.should_retrain = should_retrain

    def get_data(self, period="max", interval="1d"):
        interval = "1m" if self.args.min else "1d"
        self.raw_data = yf.download(self.ticker, period=period, interval=interval, progress=False)
        data = self.raw_data[["Close"]]
        data.reset_index(inplace=True)
        data.set_index("Date" if not self.args.min else "Datetime", inplace=True)
        return data

    def create_x_y_split(self):
        data = self.scaler.fit_transform(self.data)
        dataX, dataY = [], []
        for i in range(len(data) - 1):
            a = data[i : i + 1, 0]
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
        model = Sequential([
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
        ])
        model.compile(optimizer=Adam(0.001), loss="mse")
        return model

    def train(self, X_train, y_train):
        callbacks = [TqdmCallback(verbose=1)] if self.should_retrain else []
        self.model.fit(
            X_train,
            y_train,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            verbose=0,
            callbacks=callbacks,
        )
        
    def load_weights(self):
        if self.args.weights is not None:
            if os.path.isfile(self.args.weights):
                print(f"Used model weights from '{self.args.weights}'")
                self.model.load_weights(self.args.weights)
        else:
            path = os.path.join(get_dafault_bw_path(), "weights", self.ticker, "*.h5")
            files = glob.glob(path)
            if os.path.isfile(files[-1]):
                print(f"Used model weights from '{files[-1]}'")
                self.model.load_weights(files[-1])

    def parse_args(self):
        argparser = argparse.ArgumentParser(description="Cryptocurrency Forecast")
        argparser.add_argument("--coin", type=str, default="LTC-EUR")
        argparser.add_argument("--batch_size", type=int, default=1024)
        argparser.add_argument("--epochs", type=int, default=200)
        argparser.add_argument("--folds", type=int, default=6)
        argparser.add_argument("--retrain", action="store_true")
        argparser.add_argument("--path", type=str, default=get_dafault_bw_path())
        argparser.add_argument("--weights", type=str, default=None)
        argparser.add_argument("--future", type=int, default=7)
        argparser.add_argument("--min", action="store_true")
        args = argparser.parse_args()
        return args

    def load_history(self):
        all_train_pred = pd.DataFrame()
        all_actuals_df = pd.DataFrame()

        kf = KFold(n_splits=self.args.folds, shuffle=False)

        if not self.should_retrain:
            self.load_weights()

        def train_fold(train_index, test_index):
            X_train, X_test, y_train, y_test = self.split(
                self.X, self.y, train_index, test_index
            )
            if self.should_retrain:
                self.train(X_train, y_train)
            pred = self.scaler.inverse_transform(self.model.predict(X_test, verbose=0))

            idx = self.data.index[test_index].to_pydatetime()
            train_pred_fold = pd.DataFrame(pred, index=idx, columns=["Prediction"])
            actuals = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            actuals_fold = pd.DataFrame(actuals, index=idx, columns=["Actual"])
            return train_pred_fold, actuals_fold

        with ThreadPoolExecutor(max_workers=self.args.folds) as executor:
            futures = [
                executor.submit(train_fold, train_index, test_index)
                for train_index, test_index in kf.split(self.X)
            ]
            for future in as_completed(futures):
                train_pred_fold, actuals_fold = future.result()
                all_train_pred = pd.concat([all_train_pred, train_pred_fold])
                all_actuals_df = pd.concat([all_actuals_df, actuals_fold])

        if self.should_retrain:
            self.model.save_weights(self.weight_path)
            print(f"Saved model weights to '{self.weight_path}'")

        return all_train_pred, all_actuals_df

    def predict_future(self):
        future_predictions = []
        last_window = self.X[-1]

        time_increment = pd.Timedelta(minutes=1) if self.args.min else pd.Timedelta(days=1)
        for _ in range(self.future_days + 1):
            next_day_prediction = self.model.predict(np.array([last_window]), verbose=0)
            future_predictions.append(next_day_prediction[0])
            last_window = np.roll(last_window, -1)
            last_window[-1] = next_day_prediction

        future_predictions = self.scaler.inverse_transform(future_predictions)
        start_date = self.raw_data.index[-1]
        end_date = start_date + time_increment * self.future_days
        date_range = pd.date_range(start=start_date, end=end_date, freq=time_increment)
        future_predictions = pd.DataFrame(future_predictions, index=date_range, columns=["Prediction"])
        self.forecast_data = future_predictions

        self.save_prediction()

    def visualize(self):
        plot(self)

    def save_prediction(self):
        filepath = create_cloud_path(self.args.path, ticker=self.ticker, typeof="forecasts", filetype="csv")
        self.forecast_data.to_csv(filepath, index=True)