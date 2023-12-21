# Crypto Forecast

This Python script is designed for predicting cryptocurrency prices using a deep learning model. The model combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units. It utilizes financial data from Yahoo Finance and is built using Keras and TensorFlow.

## Features
- Hybrid CNN-RNN Architecture: Combines 1D CNN and Bidirectional LSTM layers for efficient feature extraction and sequence modeling.
- Customizable Predictions: Allows predictions for different cryptocurrencies.
- KFold Cross-Validation: Implements KFold cross-validation for training and evaluating the model.
- Data Normalization: Normalizes data for better model performance.
- Visualization: Includes functionality to plot predicted vs. actual prices.

## Usage

Run the script from the command line. You can specify the cryptocurrency, batch size, number of epochs, and the prediction length (in days). For example:

```bash 
python forecast.py --coin btc --batch_size 32 --epochs 100 --prediction 60
```

This command will predict the price of Bitcoin (btc) for the next 60 days using a batch size of 32 and 100 epochs for training.

In addition to predicting future prices, the script can also be used to test the model's performance on historical data. This feature allows you to compare actual vs. predicted prices over a all historical prices, providing insights into the model's accuracy and reliability.

To enable this mode, set the `--prediction` parameter to `-1` or just remove `--prediction` from the CLI. This will instruct the model to use the entire test dataset for making predictions and then plot these predictions against the actual historical prices.

Example command for testing historical price predictions:

```bash 
python forecast.py --coin btc --batch_size 32 --epochs 100
```
This command will train the model on historical Bitcoin (btc) data and then plot the model's predictions against the actual prices in the test dataset, allowing for a visual comparison of the model's predictive accuracy.

Remember, while this feature is useful for assessing the model's performance, it should be noted that past performance is not always indicative of future results, especially in the volatile cryptocurrency market.

### Parameters
- --coin: Cryptocurrency symbol (default: eth for Ethereum).
- --batch_size: Size of batches used in training (default: 32).
- --epochs: Number of epochs for training the model (default: 100).
- --prediction: Number of days to predict. Use -1 for full test data prediction (default: -1).

## Model Architecture
- Input Layer: Conv1D layer for sequence feature extraction.
- Hidden Layers: Two Bidirectional LSTM layers for temporal data processing.
- Regularization: Dropout and L2 regularization to prevent overfitting.
- Output Layer: Dense layer for final price prediction.
