import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# https://github.com/ShrutiAppiah/crypto-forecasting-with-neuralnetworks/

dataset_path = 'data/all_bitcoin.csv'
batch_size = 64 
epochs = 100
duration = 50

def main(coin, batch_size, epochs, duration):

    if coin == 'BIT': dataset_path = 'data/all_bitcoin.csv'
    elif coin == 'ETH': dataset_path = 'data/all_eth.csv'
    else: raise ValueError('Invalid crypto type, use BIT or ETH')

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
    data = data.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Market Cap'], axis=1)
    data = data.values

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Prepare the X and Y label
    X, y = create_dataset(data, duration)

    # Take 80% of data as the training sample and 20% as testing sample
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    # Build a Keras Sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

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

    assert acc * 100 <= 100 and acc * 100 >= 0, f"Accuracy {round(acc * 100)}% is not within range 0-100%"

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

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--bit", action="store_true", default=True, help="Use Bitcoin dataset")
    argparser.add_argument("--eth", action="store_true", default=False, help="Use Ethereum dataset")
    argparser.add_argument('--batch_size', type=int, default=batch_size, help='batch size for training')
    argparser.add_argument('--epochs', type=int, default=epochs, help='number of epochs for training')
    argparser.add_argument('--duration', type=int, default=duration, help='Forecast Horizon')
    parse = argparser.parse_args()

    if parse.eth: coin = "ETH"
    else: coin = "BIT"

    main(coin, batch_size=parse.batch_size, epochs=parse.epochs, duration=parse.duration)

