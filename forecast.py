import yfinance as yf
import argparse
import numpy as np
from sklearn.model_selection import KFold
from keras import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Bidirectional, Dropout
from keras.regularizers import l2, l1_l2
from keras.optimizers.legacy import Adam

from utils import load_and_preprocess_data, normalize_data, create_dataset, plot_predictions

def build_and_compile_model(num_features):
    """ Build and compile the Keras Sequential model. """
    model = Sequential([
        Conv1D(64, 1, activation="relu", input_shape=(1, num_features)),
        Bidirectional(LSTM(50, activation="relu", return_sequences=True)),
        Bidirectional(LSTM(50, activation="relu", return_sequences=True)),
        Dropout(0.2),
        Flatten(),
        Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return model

def main(coin, batch_size, epochs, prediction):

    data = load_and_preprocess_data(coin)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)

    kf = KFold(n_splits=5, shuffle=False)
    all_test_predictions = []
    all_test_actuals = []

    if prediction == -1:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = build_and_compile_model(X.shape[2])
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            testPredict = scaler.inverse_transform(model.predict(X_test))
            testY = scaler.inverse_transform(y_test.reshape(-1, 1))
            all_test_predictions.extend(testPredict)
            all_test_actuals.extend(testY)
        plot_predictions(np.concatenate(all_test_predictions), coin, testY=all_test_actuals)
    else:
        for train_index, _ in kf.split(X):
            X_train, y_train = X[train_index], y[train_index]
            model = build_and_compile_model(X.shape[2])
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            testPredict = scaler.inverse_transform(model.predict(X))
            all_test_predictions.extend(testPredict)
        plot_predictions(np.concatenate(all_test_predictions)[-prediction:], coin)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction")
    argparser.add_argument("--coin", type=str, default="eth")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--prediction", type=int, default=-1)
    args = argparser.parse_args()

    main(
        coin=f"{args.coin.upper()}-USD", 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        prediction=args.prediction,
    )