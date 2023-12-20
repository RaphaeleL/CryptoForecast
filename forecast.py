import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from hyperparameter import batch_sizes, epochs, dataset_paths, durations, choices, dataset_names
from keras import Sequential
from keras.layers import Dense, LSTM

def load_and_preprocess_data(path, dataset_name):
    """ Load and preprocess data from the given path. """
    try:
        if dataset_name == "investing":
            to_drop = ["Datum", "Eröffn.", "Hoch", "Tief", "Vol.", "+/- %"]
            data = pd.read_csv(path)
        elif dataset_name == "coinmarketcap":
            to_drop = ["timeOpen", "timeClose", "timeHigh", "timeLow", "name", "open", "high", "low", "volume", "marketCap", "timestamp"]
            data = pd.read_csv(path, sep=";")
        for column in to_drop:
            if column in data.columns:
                data = data.drop(column, axis=1)
        data = data.replace("\.", "", regex=True).replace(",", ".", regex=True).astype(float)
        data = data.iloc[::-1] 
        return data
    except FileNotFoundError:
        print(f"File not found: {path}")
        exit(1)

def normalize_data(data):
    """ Normalize data using Min-Max Scaler. """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)

def create_dataset(dataset, duration):
    """ Create dataset matrix for training and testing. """
    dataX, dataY = [], []
    for i in range(len(dataset) - duration):
        a = dataset[i:(i + duration), 0]
        dataX.append(a)
        dataY.append(dataset[i + duration, 0])
    return np.array(dataX).reshape(-1, duration, 1), np.array(dataY)

def build_and_compile_model(duration, num_features):
    """ Build and compile the Keras Sequential model. """
    model = Sequential([
        LSTM(100, activation="relu", input_shape=(duration, num_features), return_sequences=True),
        LSTM(100, activation="relu", return_sequences=False),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def plot_predictions(testPredict, coin, dataset_name, testY=None, ):
    """ Plot the actual vs predicted prices. """
    if testY is not None: plt.plot(testY, label="Actual Price")
    title = f"{coin} Price Prediction \n{dataset_name}"
    plt.plot(testPredict, label="Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Price in $")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def print_predictions(testPredict, coin, dataset_name):
    """ Print the actual vs predicted prices. """
    print(f"Predictions for {coin} using {dataset_name}:")
    for key, value in enumerate(testPredict):
        print(f"Day {key}: ${value[0]:.2f}")

def main(coin, batch_size, epochs, duration, dataset_path, dataset_name, real_pred=False, print_pred=False):
    data = load_and_preprocess_data(dataset_path, dataset_name)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data, duration)
    if not real_pred:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
        model = build_and_compile_model(duration, X_train.shape[2])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        testPredict = scaler.inverse_transform(model.predict(X_test))
        testY = scaler.inverse_transform(y_test.reshape(-1, 1))
        plot_predictions(testPredict[-duration + 1:], coin, dataset_name, testY=testY[-duration:])
    else: 
        model = build_and_compile_model(duration, X.shape[2])
        model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)
        testPredict = scaler.inverse_transform(model.predict(X))
        if print_pred: print_predictions(testPredict[-duration + 1:], coin, dataset_name)
        plot_predictions(testPredict[-duration + 1:], coin, dataset_name)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction")
    argparser.add_argument("--coin", type=str, choices=choices, required=True, help=f"Specify the cryptocurrency: {choices}")
    argparser.add_argument("--dataset_name", type=str, choices=dataset_names, required=True, help=f"Specify the dataset: {choices}")
    argparser.add_argument("--batch_size", type=int, help="Batch size for training")
    argparser.add_argument("--epochs", type=int, help="Number of epochs for training")
    argparser.add_argument("--duration", type=int, help="Forecast Horizon")
    argparser.add_argument("--real_prediction", action="store_true", help="Real prediction")
    argparser.add_argument("--print_result", action="store_true", help="Print the Result of the Prediction")
    args = argparser.parse_args()

    main(
        coin=args.coin.upper(), 
        batch_size=batch_sizes[args.coin] if not args.batch_size else args.batch_size, 
        epochs=epochs[args.coin] if not args.epochs else args.epochs, 
        duration=durations[args.coin] if not args.duration else args.duration,
        dataset_path=dataset_paths[args.coin][args.dataset_name],
        dataset_name=args.dataset_name,
        real_pred=args.real_prediction,
        print_pred=args.print_result
    )