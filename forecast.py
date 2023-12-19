import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from hyperparameter import batch_sizes, epochs, dataset_paths, durations, choices
from keras import Sequential
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed

# https://de.investing.com/crypto/currencies

def load_and_preprocess_data(path, to_drop=["Datum", "Eröffn.", "Hoch", "Tief", "Vol.", "+/- %"]):
    """ Load and preprocess data from the given path. """
    try:
        data = pd.read_csv(path)
        data = data.drop(to_drop, axis=1)
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

def calculate_accuracy(testY, testPredict):
    """ Calculate the accuracy of the model. """
    acc = 0
    for index, element in enumerate(testY):
        acc += 1 - abs((element - testPredict[index])[0]) / element[0]
    acc /= len(testY)
    return round(acc * 100, 1)

def plot_predictions(testPredict, duration, coin, testY=None):
    """ Plot the actual vs predicted prices. """
    if testY is not None:
        acc = calculate_accuracy(testY, testPredict)
        title = f"{coin} Price Prediction with a {acc}% Accuracy"
        plt.plot(testY[-duration:], label="Actual Price")
    else: 
        title = f"{coin} Price Prediction (real)"
    plt.plot(testPredict[-duration + 1:], label="Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Price in $")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def main(coin, batch_size, epochs, duration, dataset_path, real_prediction=False):
    data = load_and_preprocess_data(dataset_path)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data, duration)
    if not real_prediction:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
        model = build_and_compile_model(duration, X_train.shape[2])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        testPredict = scaler.inverse_transform(model.predict(X_test))
        testY = scaler.inverse_transform(y_test.reshape(-1, 1))
        plot_predictions(testPredict, duration, coin, testY=testY)
    else: 
        model = build_and_compile_model(duration, X.shape[2])
        model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)
        testPredict = scaler.inverse_transform(model.predict(X))
        for key, value in enumerate(testPredict[-duration:]):
            print(f"Day {key}: ${value[0]:.2f}")
        plot_predictions(testPredict, duration, coin)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction")
    argparser.add_argument("--coin", type=str, choices=choices, required=True, help=f"Specify the cryptocurrency: {choices}")
    argparser.add_argument("--batch_size", type=int, help="Batch size for training")
    argparser.add_argument("--epochs", type=int, help="Number of epochs for training")
    argparser.add_argument("--duration", type=int, help="Forecast Horizon")
    argparser.add_argument("--real_prediction", action="store_true", help="Real prediction")
    args = argparser.parse_args()

    main(
        coin=args.coin.upper(), 
        batch_size=batch_sizes[args.coin] if not args.batch_size else args.batch_size, 
        epochs=epochs[args.coin] if not args.epochs else args.epochs, 
        duration=durations[args.coin] if not args.duration else args.duration,
        dataset_path=dataset_paths[args.coin],
        real_prediction=args.real_prediction
    )