import threading
import pandas as pd
from sklearn.model_selection import KFold

from utils import (
    load_and_preprocess_data,
    normalize_data,
    create_dataset,
    plot,
    plot_all_agents,
    split, 
    train,
    argument_parser,
    check
)

def crypto_forecast(coin, batch_size, epochs, agent, agents, folds, reverse, test_pred, test_actu, real_pred):
    data = load_and_preprocess_data(coin, reverse)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)
    kf = KFold(n_splits=folds, shuffle=False)

    predictions = []
    actuals = []
    real_predictions = []

    for index, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Training Agent {agent+1}/{agents} for Fold {index+1}/{folds}")
        X_train, X_test, y_train, y_test = split(X, y, train_index, test_index)
        model = train(X, y, X_train, y_train, batch_size, epochs)
        prediction = scaler.inverse_transform(model.predict(X_test))
        predictions.extend(prediction)
        actuals.extend(scaler.inverse_transform(y_test.reshape(-1, 1)))

    test_pred.append(predictions)
    test_actu.append(actuals)

    for index, (train_index, _) in enumerate(kf.split(X)):
        print(f"Training Agent {agent+1}/{agents} for Fold {index+1}/{folds}")
        X_train, y_train = split(X, y, train_index) 
        model = train(X, y, X_train, y_train, batch_size, epochs)
        prediction = scaler.inverse_transform(model.predict(X[-10:]))
        real_predictions.extend(prediction)

    starting_index = len(test_pred[0]) + 1
    real_pred_indices = list(range(starting_index, starting_index + len(real_predictions)))

    real_pred.append(pd.Series(real_predictions, index=real_pred_indices))

    if not check(test_actu[-1][-1], real_pred[0][starting_index]):
        print("*****************************************************")
        print(f"***** ERROR (A{agent+1}): Prediction is not accurate enough.")
        print(f"***** Actual Value: {test_actu[-1][-1]}")
        print(f"***** Predicted:: {real_pred[0][starting_index]}")
        print("*****************************************************")
        # TODO - Add a way to retrain the model

    return test_pred, test_actu, real_pred

if __name__ == "__main__":
    args = argument_parser()

    if args.plot_coin:
        data = load_and_preprocess_data(args.coin, args.reverse)
        plot(args.coin, data, exit_after=True)
        exit()

    threads, test_pred, test_actu, real_pred = [], [], [], []

    for agent in range(args.agents):
        thread = threading.Thread(
            target=crypto_forecast,
            args=(args.coin, args.batch_size, args.epochs, agent, args.agents, args.folds,
                args.reverse, test_pred, test_actu, real_pred),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    plot_all_agents(args.coin, test_pred, test_actu, real_pred)
