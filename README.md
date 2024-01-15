# Crypto Forecast

This Python script is designed for predicting cryptocurrency prices using a deep learning model. The model combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units. It utilizes financial data from Yahoo Finance and is built using Keras and TensorFlow.

## Features

- Hybrid CNN-RNN Architecture: Combines 1D CNN and Bidirectional LSTM layers for efficient feature extraction and sequence modeling.
- Customizable Predictions: Allows predictions for different cryptocurrencies.
- KFold Cross-Validation: Implements KFold cross-validation for training and evaluating the model.
- Data Normalization: Normalizes data for better model performance.
- Visualization: Includes functionality to plot predicted vs. actual prices.

## Usage

Run the script from the command line. You can specify the cryptocurrency, batch size, number of epochs, and the prediction length (in days), but you should be fine with using the defaults and just run `python3 forecast.py`. Nevertheless, an example:

```bash 
python forecast.py --coin ETH-USD
```

If you want to Retrain the Weights of the Coin, or want to create a new Weight File for a new Ticker, then just append `--retrain`. 

### Parameters

- `--coin`: Cryptocurrency symbol (default: eth for Ethereum).
- `--batch_size`: Size of batches used in training (default: 32).
- `--epochs`: Number of epochs for training the model (default: 100).
- `--folds`: Number of Folds for the KFold (default: 5).
- `--prediction`: Number of days to predict. Use -1 for full test data prediction (default: -1).
- `--retrain`: (Re-) Train the Ticker. 
- `--debug`: Predict the History with Weights and see how the Model is working.
- `--auto`: Use all Available Tickers (for those with a Weight File) and Predict the Future for them.
- `--path`: Location of the `forecast/` and `metrics/` folder, aka. the Results of the Prediction (default: '~/bwSyncShare/PadWise-Trading')

## Model Architecture

- Input Layer: Conv1D Layer for sequence feature extraction.
- Hidden Layers: Three Bidirectional LSTM Layer for temporal data processing.
- Regularization: Dropouts and L2 regularization to prevent overfitting.
- Output Layer: Dense layer for final price prediction.
