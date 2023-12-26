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
python forecast.py --coin ETH-USD --batch_size 32 --epochs 20 --agents 4 --folds 6 --prediction 7 
```

This command will predict the price of ETH for the next 7 days by 4 Agents, but it will just show you the best Agent. If you want to see all Agent's Prediction, just append `--show_all`.

### Parameters

- `--coin`: Cryptocurrency symbol (default: eth for Ethereum).
- `--batch_size`: Size of batches used in training (default: 32).
- `--epochs`: Number of epochs for training the model (default: 100).
- `--agents`: Number of agents to predict parallel (default: 1).
- `--folds`: Number of Folds for the KFold (default: 5).
- `--prediction`: Number of days to predict. Use -1 for full test data prediction (default: -1).
- `--show_all`: Show's the Result of all Agents. 

## Model Architecture

- Input Layer: Conv1D layer for sequence feature extraction.
- Hidden Layers: Two Bidirectional LSTM layers for temporal data processing.
- Regularization: Dropout and L2 regularization to prevent overfitting.
- Output Layer: Dense layer for final price prediction.
