#!/usr/bin/env python3

from forecast import CryptoForecast
from utils import plot


if __name__ == "__main__":

    cryptoforecast = CryptoForecast()

    # Step 0: (Re-)Train the Neural Network
    if cryptoforecast.args.retrain:
        cryptoforecast.load_history()
        exit()

    # Step 1: Load the History
    cryptoforecast.load_history()
    # Step 2: Predict the future
    prediction = cryptoforecast.predict_future()
    plot(cryptoforecast, prediction)
    # Step 3: Validate the Predictions with a Trend Analysis
    # Step 4: Validate the Predictions with LLMs
