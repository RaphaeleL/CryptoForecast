#!/usr/bin/env python3

import tensorflow as tf
import threading
import os
import warnings
from sklearn.model_selection import KFold
from utils import argument_parser, load_history, predict_future, plot, \
        load_and_preprocess_data, normalize_data, create_dataset, \
        evaluate_agent_performance, select_best_agent, performance_output

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def crypto_forecast(data, scaler, X, y, f, agent, args, train, test, val):
    """Forecast cryptocurrency prices."""
    train, test = load_history(args, f, agent, X, y, scaler, data, train, test)
    val = predict_future(args, kf, agent, X, y, scaler, data, val)
    return train, test, val


def main(coin, data, scaler, X, y, kf, args):
    threads, train, test, val = [], [], [], []

    for agent in range(args.agents):
        thread = threading.Thread(
            target=crypto_forecast,
            args=(data, scaler, X, y, kf, agent, args, train, test, val),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if args.debug > 1:
        performance_data = evaluate_agent_performance(test, train)
        best_agent = select_best_agent(performance_data)
        p_change = performance_output(args, val, best_agent, coin)
        mae_score = evaluate_agent_performance(test, train)[best_agent]
        plot(args.coin, best_agent, val, args.prediction, mae_score, p_change)


if __name__ == "__main__":
    args = argument_parser()
    data = load_and_preprocess_data(args.coin)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)
    kf = KFold(n_splits=args.folds, shuffle=False)

    main(args.coin, data, scaler, X, y, kf, args)
