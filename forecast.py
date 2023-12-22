import argparse
from sklearn.model_selection import KFold
from keras import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Bidirectional, Dropout
from keras.regularizers import l2
from keras.optimizers.legacy import Adam
from tqdm.keras import TqdmCallback

from utils import load_and_preprocess_data, normalize_data, create_dataset, plot, plot_all_agents

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
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def main(coin, batch_size, epochs, prediction, agents, folds, plot_coin, reverse):
    all_agent_test_predictions = []
    all_agent_test_actuals = []
    for agent in range(agents):
        data = load_and_preprocess_data(coin, reverse)
        if plot_coin: plot(coin, data, exit_after=True)
        scaler, normalized_data = normalize_data(data)
        X, y = create_dataset(normalized_data)

        kf = KFold(n_splits=folds, shuffle=False)
        all_test_predictions = []
        all_test_actuals = []

        if prediction == -1:
            for index, (train_index, test_index) in enumerate(kf.split(X)):
                print(f"Training Agent {agent+1}/{agents} for Fold {index+1}/{folds}")
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model = build_and_compile_model(X.shape[2])
                model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[TqdmCallback(verbose=2)])
                testPredict = scaler.inverse_transform(model.predict(X_test))
                testY = scaler.inverse_transform(y_test.reshape(-1, 1))
                all_test_predictions.extend(testPredict)
                all_test_actuals.extend(testY)
        else:
            for index, (train_index, _) in enumerate(kf.split(X)):
                print(f"Training Agent {agent+1}/{agents} for Fold {index+1}/{folds}")
                X_train, y_train = X[train_index], y[train_index]
                model = build_and_compile_model(X.shape[2])
                model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[TqdmCallback(verbose=2)])
                testPredict = scaler.inverse_transform(model.predict(X[-prediction:]))
                all_test_predictions.extend(testPredict[-prediction:])
        all_agent_test_predictions.append(all_test_predictions)
        all_agent_test_actuals.append(all_test_actuals)
    plot_all_agents(all_agent_test_predictions, coin, all_agent_test_actuals if prediction == -1 else None)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction")
    argparser.add_argument("--coin", type=str, default="eth")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--prediction", type=int, default=-1)
    argparser.add_argument("--agents", type=int, default=1)
    argparser.add_argument("--folds", type=int, default=5)
    argparser.add_argument("--plot_coin", action="store_true")
    argparser.add_argument("--reverse", action="store_true")
    args = argparser.parse_args()

    main(
        coin=f"{args.coin.upper()}", 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        prediction=args.prediction,
        agents=args.agents,
        folds=args.folds,
        plot_coin=args.plot_coin,
        reverse=args.reverse
    )