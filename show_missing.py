import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(path, dataset_name):
    """ Load and preprocess data from the given path. """
    try:
        to_drop = ["timeOpen", "timeClose", "timeHigh", "timeLow", "name", "open", "high", "low", "volume", "marketCap", "timestamp"]
        data = pd.read_csv(path, sep=";")
        for column in to_drop:
            if column in data.columns:
                data = data.drop(column, axis=1)
        data = data.replace("\.", "", regex=True).replace(",", ".", regex=True).astype(float)
        data = data.iloc[::-1].reset_index(drop=True)
        return data
    except FileNotFoundError:
        print(f"File not found: {path}")
        exit(1)


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

if __name__ == "__main__":
    data = load_and_preprocess_data("data/ethereum_coinmarketcap_10-19_Dez_2023.csv", "10 - 20 Dez. 2023 ETH")
    plot_predictions(data, "ETH", "10 - 20 Dez. 2023 ETH")