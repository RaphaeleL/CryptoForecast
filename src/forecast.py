import os
import glob
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers.legacy import Adam
from tqdm.keras import TqdmCallback
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import plot, create_cloud_path, get_allowed_coins, get_default_coin
from src.models import bitcoin, default_model


class CryptoForecast:
    def __init__(self, epochs, batch_size, ticker, folds, retrain, path, weights, future_days):
        self.epochs = epochs
        self.batch_size = batch_size
        self.ticker = ticker
        self.folds = folds
        self.retrain = retrain
        self.path = path
        self.weights = weights
        self.future_days = future_days

        self.weight_path = create_cloud_path(self.path, ticker=self.ticker, typeof="weights", filetype="h5")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.X = None
        self.Y = None
        self.raw_data = None
        self.forecast_data = []

        self.cut_out_data = None
        self.losses = []
        self.predict_count = 0

    def preprocess(self):
        self.get_data()
        self.create_x_y_split()
        self.build_model()

    def postprocess(self):
        self.forecast_data["Prediction"] = self.forecast_data["Prediction"].apply(lambda x: "{:.2f}".format(x))
        self.forecast_data["Prediction"] = self.forecast_data["Prediction"].astype(float)
        self.forecast_data = self.forecast_data[["Prediction"]]

    def get_data(self, period="max", interval="1d"):
        self.raw_data = yf.download(self.ticker, period=period, interval=interval, progress=False)
        self.data = self.raw_data[["Close"]]
        self.data.reset_index(inplace=True)
        self.data.set_index("Date", inplace=True)
        self.cut_out_data = self.data[-self.future_days:]
        self.data = self.data[:-self.future_days + 1]
        self.future_days *= 2

        self.raw_data = self.raw_data[["Close"]]
        self.raw_data.reset_index(inplace=True)
        self.raw_data.set_index("Date", inplace=True)

    def create_x_y_split(self):
        data = self.scaler.fit_transform(self.data)
        dataX, dataY = [], []
        for i in range(len(data) - 1):
            a = data[i: i + 1, 0]
            dataX.append(a)
            dataY.append(data[i + 1, 0])
        self.X, self.y = np.array(dataX).reshape(-1, 1, 1), np.array(dataY)

    def split(self, X, y, train_index, test_index):
        X_train = X[train_index]
        y_train = y[train_index]
        if test_index is not None:
            X_test = X[test_index]
            y_test = y[test_index]
            return X_train, X_test, y_train, y_test
        return X_train, y_train

    def build_model(self):
        self.model = default_model(self)
        if any(ticker in self.ticker for ticker in get_allowed_coins()):
            # NOTE: All allowed coins are using the bitcoin model, because it 
            #       is the best one. It may be better to implement a model for 
            #       each coin. 
            self.model = bitcoin(self)
        self.model.compile(optimizer=Adam(0.001), loss="mse")

    def train(self, X_train, y_train):
        callbacks = [TqdmCallback(verbose=1)] if self.retrain else []
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            callbacks=callbacks,
        )
        self.losses.append(history.history["loss"][-1])

    def load_weights(self):
        if self.weights is not None:
            if os.path.isfile(self.weights):
                print(f"Used model weights from '{self.weights}'")
                self.model.load_weights(self.weights)
        else:
            path = os.path.join(self.path, "weights", self.ticker, "*.h5")
            files = glob.glob(path)
            if len(files) == 0:
                print(f"No weights found in '{path.replace('*.h5', '')}', trying to load default weights from {get_default_coin()}.")
                path = os.path.join(self.path, "weights", get_default_coin(), "*.h5")
                files = glob.glob(path)
            assert len(files) > 0, f"No weights found in '{path.replace('*.h5', '')}', please train your ticker ({self.ticker}) under your choosen path."
            if os.path.isfile(files[-1]):
                print(f"Used model weights from '{files[-1]}'")
                self.model.load_weights(files[-1])

    def load_history(self):
        all_train_pred = pd.DataFrame()
        all_actuals_df = pd.DataFrame()

        tss = TimeSeriesSplit(n_splits=self.folds)

        if not self.retrain:
            self.load_weights()

        def train_fold(train_index, test_index):
            X_train, X_test, y_train, y_test = self.split(self.X, self.y, train_index, test_index)
            if self.retrain:
                self.train(X_train, y_train)
            pred = self.scaler.inverse_transform(self.model.predict(X_test, verbose=0))

            idx = self.data.index[test_index].to_pydatetime()
            train_pred_fold = pd.DataFrame(pred, index=idx, columns=["Prediction"])
            actuals = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            actuals_fold = pd.DataFrame(actuals, index=idx, columns=["Actual"])
            return train_pred_fold, actuals_fold

        with ThreadPoolExecutor(max_workers=self.folds) as executor:
            futures = [
                executor.submit(train_fold, train_index, test_index)
                for train_index, test_index in tss.split(self.X)
            ]
            for future in as_completed(futures):
                train_pred_fold, actuals_fold = future.result()
                all_train_pred = pd.concat([all_train_pred, train_pred_fold])
                all_actuals_df = pd.concat([all_actuals_df, actuals_fold])

        if self.retrain:
            self.model.save_weights(self.weight_path)
            print(f"Saved model weights to '{self.weight_path}'")

        return all_train_pred, all_actuals_df

    def predict_future(self):
        input_window_size = self.future_days
        self.X = self.X.values if isinstance(self.X, pd.DataFrame) else self.X
        future_input = self.X[-input_window_size:]
        future_prediction = self.model.predict(future_input, verbose=0)
        prediction = self.scaler.inverse_transform(future_prediction)
        next_day = self.data.index[-2] + pd.Timedelta(days=1)
        future_dates = pd.date_range(start=next_day, periods=input_window_size, freq="D")
        res = pd.DataFrame(prediction, index=future_dates, columns=["Prediction"])
        self.forecast_data = res
        self.save_prediction()

    def visualize(self):
        for i, (index, value) in enumerate(self.forecast_data.iterrows()):
            print(f"Predicted Price for {index.strftime('%d. %b %Y')}: ", end="")
            print(f"{value[0]} {self.ticker.split('-')[1] if '-' in self.ticker else ''} ", end="")
            print(f"Day: {str(i+1)}")
        plot(self)

    def save_prediction(self):
        filepath = create_cloud_path(self.path, ticker=self.ticker, typeof="forecasts", filetype="csv")
        self.forecast_data.to_csv(filepath, index=True, sep=";")
