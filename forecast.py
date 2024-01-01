#!/usr/bin/env python3

import threading, os, warnings
from sklearn.model_selection import KFold
from utils import *

warnings.filterwarnings('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  

def crypto_forecast(data, scaler, X, y, kf, agent, args, test_pred, test_actu, real_pred, prediction_days=1):
    """Forecast cryptocurrency prices."""
    test_pred, test_actu = load_history(args, kf, agent, X, y, scaler, data, test_pred, test_actu)
    real_pred = predict_future(args, kf, agent, X, y, scaler, data, real_pred, prediction_days)
    return test_pred, test_actu, real_pred

def main(coin, data, scaler, X, y, kf, args):
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

    if args.debug > 1:
        performance_data = evaluate_agent_performance(test_actu, test_pred)
        best_agent = select_best_agent(performance_data)
        percentage_change =  performance_output(args, real_pred, best_agent, coin)
        mae_score = evaluate_agent_performance(test_actu, test_pred)[best_agent]
        plot(args.coin, best_agent, real_pred, args.prediction, mae_score, percentage_change)

if __name__ == "__main__":
    args = argument_parser()
    data = load_and_preprocess_data(args.coin)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)
    kf = KFold(n_splits=args.folds, shuffle=False)

    main(args.coin, data, scaler, X, y, kf, args)