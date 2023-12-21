import argparse
import matplotlib.pyplot as plt
from forecast import load_and_preprocess_data
from hyperparameter import choices, dataset_names 

def plot_predictions(testPredict, testY=None):
    """ Plot the actual vs predicted prices. """
    if testY is not None: plt.plot(testY)
    plt.plot(testPredict)
    plt.xlabel("Days")
    plt.ylabel("Price in $")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Data Plotter")
    argparser.add_argument("--dataset", type=str, required=True, help=f"Specify the dataset: {choices}")
    args = argparser.parse_args()

    path = args.dataset if args.dataset else "data/ethereum_coinmarketcap_10-19_Dez_2023.csv"
    data = load_and_preprocess_data(path)
    plot_predictions(data)