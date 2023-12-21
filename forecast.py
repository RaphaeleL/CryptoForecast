import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Bidirectional, Dropout, GRU, TimeDistributed, MaxPool1D, BatchNormalization
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers.legacy import SGD, Adam

from hyperparameter import dataset_paths, choices, dataset_names

def load_and_preprocess_data(path):
    """ Load and preprocess data from the given path. """
    try:
        if "investing" in path:
            to_drop = ["Datum", "Eröffn.", "Hoch", "Tief", "Vol.", "+/- %"]
            data = pd.read_csv(path)
        elif "coinmarketcap" in path:
            to_drop = ["timeOpen", "timeClose", "timeHigh", "timeLow", "name", "open", "high", "low", "volume", "marketCap", "timestamp"]
            data = pd.read_csv(path, sep=";")
        elif "yahoo" in path:
            to_drop = ["Date", "Open", "High", "Low", "Adj Close", "Volume"]
            data = pd.read_csv(path)
        elif "gecko" in path:
            to_drop = ["snapped_at", "market_cap", "total_volume"]
            data = pd.read_csv(path)
        for column in to_drop:
            if column in data.columns:
                data = data.drop(column, axis=1)
        data = data.replace("\.", "", regex=True).replace(",", ".", regex=True).astype(float)
        data = data.iloc[::-1].reset_index(drop=True)
        return data
    except FileNotFoundError:
        print(f"File not found: {path}")

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

def build_and_compile_model(num_features):
    """ Build and compile the Keras Sequential model. """

    model = Sequential([
        Conv1D(64, 1, activation="relu", input_shape=(1, num_features)),
        Bidirectional(LSTM(50, activation="relu", return_sequences=True)),
        Bidirectional(LSTM(50, activation="relu", return_sequences=True)),
        Dropout(0.2),
        Flatten(),
        Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return model

def plot_predictions(testPredict, coin="n/a", reverse_plot=False, dataset_name="n/a", testY=None):
    """ Plot the actual vs predicted prices. """
    if testY is not None: plt.plot(testY[::-1], label="Actual Price")
    title = f"{coin} Price Prediction \n{dataset_name}"
    plt.plot(testPredict if not reverse_plot else testPredict[::-1], label="Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Price in $")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(coin, batch_size, epochs, dataset_path, dataset_name, real_pred, reverse_plot):
    data = load_and_preprocess_data(dataset_path)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)

    kf = KFold(n_splits=5, shuffle=False)
    all_test_predictions = []
    all_test_actuals = []

    if real_pred == -1:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = build_and_compile_model(X.shape[2])
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            testPredict = scaler.inverse_transform(model.predict(X_test))
            testY = scaler.inverse_transform(y_test.reshape(-1, 1))
            all_test_predictions.extend(testPredict)
            all_test_actuals.extend(testY)
        plot_predictions(np.concatenate(all_test_predictions), coin, reverse_plot, dataset_name, testY=all_test_actuals)
    else:
        for train_index, _ in kf.split(X):
            X_train, y_train = X[train_index], y[train_index]
            model = build_and_compile_model(X.shape[2])
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            testPredict = scaler.inverse_transform(model.predict(X))
            all_test_predictions.extend(testPredict)
        plot_predictions(np.concatenate(all_test_predictions)[-real_pred:], coin, reverse_plot, dataset_name)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction")
    argparser.add_argument("--coin", type=str, default="eth", choices=choices)
    argparser.add_argument("--dataset", type=str, default="gecko", choices=dataset_names)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--prediction", type=int, default=-1)
    argparser.add_argument("--revplot", action="store_true", default=-1)
    args = argparser.parse_args()

    main(
        coin=args.coin.upper(), 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        dataset_path=dataset_paths[args.coin][args.dataset],
        dataset_name=args.dataset,
        real_pred=args.prediction,
        reverse_plot=args.revplot
    )