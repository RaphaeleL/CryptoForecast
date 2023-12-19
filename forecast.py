import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# https://de.investing.com/crypto/currencies
# https://github.com/ShrutiAppiah/crypto-forecasting-with-neuralnetworks/

def main(coin, batch_size, epochs, duration, dataset_path):

    # Function to create a dataset matrix
    def create_dataset(dataset, duration):
        dataX, dataY = [], []
        for i in range(len(dataset) - duration):
            a = dataset[i:(i + duration), 0]
            dataX.append(a)
            dataY.append(dataset[i + duration, 0])
        return np.array(dataX), np.array(dataY)

    # Import data
    data = pd.read_csv(dataset_path)

    # Preprocess the data
    data = data.drop(["Datum", "Eröffn.", "Hoch", "Tief", "Vol.", "+/- %"], axis=1)
    data = data.replace("\.", "", regex=True)
    data = data.replace(",", ".", regex=True)
    data = data.astype(float)

    # Reverse 
    data = data.iloc[::-1]

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Prepare the X and Y label
    X, y = create_dataset(data, duration)

    # Take 80% of data as the training sample and 20% as testing sample
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    # Build a Keras Sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # Evaluate the model
    testPredict = model.predict(X_test)

    # Inverse Transform the predicted and testing data outputs to get accuracy
    testPredict = scaler.inverse_transform(testPredict)
    y_test_reshaped = y_test.reshape(-1, 1)
    testY = scaler.inverse_transform(y_test_reshaped)

    # Calculate the accuracy
    acc = 0
    for index, element in enumerate(testY):
        acc += 1 - abs((element - testPredict[index])[0]) / element[0]
    acc /= len(testY)

    print(f"Prediction: {testPredict[-1]}")
    print(f"Actual: {testY[-1]}")
    print(f"Accuracy: {round(acc * 100, 2)}%")

    # Plot baseline and predictions
    plt.plot(testY[-duration:], label="Actual Price")
    plt.plot(testPredict[-duration:], label="Predicted Price")
    plt.xlabel("Day")
    plt.ylabel("Bitcoin Price")
    plt.title(f"Price Prediction for {coin} is {round(acc * 100, 2)}% accurate")
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--bit", action="store_true", default=False, help="Use Bitcoin dataset")
    argparser.add_argument("--eth", action="store_true", default=False, help="Use Ethereum dataset")
    argparser.add_argument("--ltc", action="store_true", default=False, help="Use Litecoin dataset")
    argparser.add_argument("--batch_size", type=int, help="batch size for training")
    argparser.add_argument("--epochs", type=int, help="number of epochs for training")
    argparser.add_argument("--duration", type=int, help="Forecast Horizon")
    parse = argparser.parse_args()

    if parse.bit:
        main(
            "BIT", 
            batch_size=parse.batch_size if parse.batch_size else 32, 
            epochs=parse.epochs if parse.epochs else 5, 
            duration=parse.duration if parse.duration else 50,
            dataset_path="data/full_bitcoin.csv"
        )
    elif parse.eth: 
        main(
            "ETH", 
            batch_size=parse.batch_size if parse.batch_size else 32, 
            epochs=parse.epochs if parse.epochs else 200, 
            duration=parse.duration if parse.duration else 50,
            dataset_path="data/full_eth.csv"
        )
    elif parse.ltc: 
        main(
            "LTC", 
            batch_size=parse.batch_size if parse.batch_size else 32, 
            epochs=parse.epochs if parse.epochs else 20, 
            duration=parse.duration if parse.duration else 50,
            dataset_path="data/full_litecoin.csv"
        )
    else: 
        print("No crypto type specified. Try again with --bit, --eth or --ltc")



