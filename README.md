# Crypto Forecast

This Python script is designed for predicting cryptocurrency prices using a deep learning model. The model combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units. It utilizes financial data from Yahoo Finance and is built using Keras and TensorFlow.

## Features

- Hybrid CNN-RNN Architecture: Combines 1D CNN and Bidirectional LSTM layers for efficient feature extraction and sequence modeling.
- Customizable Predictions: Allows predictions for different cryptocurrencies.
- KFold Cross-Validation: Implements KFold cross-validation for training and evaluating the model.
- Data Normalization: Normalizes data for better model performance.
- Visualization: Includes functionality to plot predicted vs. actual prices.

## Usage

Run the script from the command line. You can specify the cryptocurrency, batch size, number of epochs, and much more, but you should be fine with using the defaults. Nevertheless, an example:

```bash 
python main.py --coin ETH-USD
```

If you want to Retrain the Weights or want to create a new Weight File, then you should use:

```bash 
python main.py --coin ETH-USD --retrain
```

### Parameters

- `--coin`: Cryptocurrency symbol (default: eth for Ethereum).
- `--batch_size`: Size of batches used in training (default: 32).
- `--epochs`: Number of epochs for training the model (default: 100).
- `--folds`: Number of Folds for the KFold (default: 5).
- `--retrain`: (Re-) Train the Ticker. 
- `--path`: Location of the `forecast/` and `plots/` folder, aka. the Results of the Prediction (default: '~/bwSyncShare/PadWise-Trading')
- `--weights`: Location of your Weight File (`weights/*.h5`), that you want to use. The default Location so save those Files is the same like in `--path`.
- `--future`: How many days do you want to predict the future (default: 30)? 
- `--min`: Forecast Minutes! The `--future` Flag indicates Minutes instead of Dates now.

## Model Architecture

- Input Layer: Conv1D Layer for sequence feature extraction.
- Hidden Layers: Three Bidirectional LSTM Layer for temporal data processing.
- Regularization: Dropouts and L2 regularization to prevent overfitting.
- Output Layer: Dense layer for final price prediction.
