#!/usr/bin/env python3

from forecast import CryptoForecast
from utils import plot


if __name__ == "__main__":

    # TODO:
    #   - Add an '--agents' CLI argument.
    #       - This will allow the user to use Agents instead of Weight Files
    #       - Number of Agents will be automatically determined by the number of cores
    #       - The Forecast Class will automalically determine if Agents or Weight Files are used
    #   - Add a 'validate()' Method to Validate the Prediction
    #       - Trend Analysis (Up, Down, Sideways, etc.)
    #       - LLMs (ChatGPT, LLaMa, etc.))
    #       - For Step 3

    cf = CryptoForecast()

    # Step 0 - (Re-)Train the Neural Network
    cf.retrain()

    # Step 1 - Load the Historical Data
    cf.load_history()

    # Step 2 - Predict the Future Price of the Ticker
    cf.predict_future()

    # Step 3 - Validate the Prediction

    # Step 5: Visualize the Prediction
    cf.visualize()
