import threading, os, warnings
import pandas as pd
from sklearn.model_selection import KFold
from utils import *

warnings.filterwarnings('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  

def crypto_forecast(data, scaler, X, y, kf, agent, args, test_pred, test_actu, real_pred, prediction_days=1):
    """Forecast cryptocurrency prices."""
    predictions = []
    actuals = []
    real_predictions = []

    # TODO: Improve Prediction Quality

    for index, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Training Agent {agent+1:02d}/{args.agents:02d} with Fold {index+1:02d}/{args.folds:02d}")
        X_train, X_test, y_train, y_test = split(X, y, train_index, test_index)
        model = train(X, X_train, y_train, args.batch_size, args.epochs)
        prediction = scaler.inverse_transform(model.predict(X_test, verbose=0))
        predictions.extend(prediction)
        actuals.extend(scaler.inverse_transform(y_test.reshape(-1, 1)))

    test_dates = data.index[test_index].to_pydatetime()
    test_pred.append(pd.DataFrame(prediction, index=test_dates, columns=['Prediction']))
    test_actu.append(pd.DataFrame(scaler.inverse_transform(y_test.reshape(-1, 1)), index=test_dates, columns=['Actual']))

    for index, (train_index, _) in enumerate(kf.split(X)):
        print(f"Predict Future for Agent {agent+1:02d}/{args.agents:02d} with Fold {index+1:02d}/{args.folds:02d}")
        X_train, y_train = split(X, y, train_index) 
        model = train(X, X_train, y_train, args.batch_size, args.epochs)
        prediction = scaler.inverse_transform(model.predict(X[-(prediction_days*12):], verbose=0))
        real_predictions.extend(prediction)

    last_day = test_actu[-1].index[-1]
    next_day = last_day + pd.Timedelta(hours=1)
    future_dates = pd.date_range(start=next_day, periods=len(real_predictions), freq='H')
    real_pred.append(pd.DataFrame(real_predictions, index=future_dates, columns=['Prediction']))

    return test_pred, test_actu, real_pred

def main(data, scaler, X, y, kf, args):
    threads, test_pred, test_actu, real_pred = [], [], [], []

    for agent in range(args.agents):
        thread = threading.Thread(
            target=crypto_forecast,
            args=(data, scaler, X, y, kf, agent, args, test_pred, test_actu, real_pred, args.prediction),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    performance_data = evaluate_agent_performance(test_actu, test_pred)
    best_agent = select_best_agent(performance_data)
    print_agent_performance_overview(args.agents, performance_data, best_agent)

    if args.show_all:
        plot_all(args.coin, best_agent, real_pred)
    plot(args.coin, best_agent, test_pred[best_agent], test_actu[best_agent], real_pred[best_agent], args.prediction)

if __name__ == "__main__":
    args = argument_parser()
    data = load_and_preprocess_data(args.coin)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)
    kf = KFold(n_splits=args.folds, shuffle=False)

    main(data, scaler, X, y, kf, args)