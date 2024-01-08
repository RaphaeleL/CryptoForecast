#!/usr/bin/env python3

from forecast import CryptoForecast
from utils import plot


if __name__ == "__main__":

    cf = CryptoForecast()

    # Step 0 - (Re-)Train the Neural Network
    cf.retrain()

    # Step 1 - Load the Historical Data
    cf.load_history()

    # Step 2 - Predict the Future Price of the Ticker
    cf.predict_future()

    # TODO: Step 3 - Validate the Predictions with a Trend Analysis
    # TODO: Step 4 - Validate the Predictions with LLMs

    # Step 5: Visualize the Prediction
    plot(cf.prediction_days, cf.forecast_data, cf.ticker)
