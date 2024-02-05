#!/usr/bin/env python3

import tensorflow as tf
import os
import time
import warnings

from src.utils import print_help, parse_args
from src.forecast import CryptoForecast

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == "__main__":
    args, argparser = parse_args()

    for coin in ["BTC-EUR", "LTC-EUR", "ETH-EUR"]:
        print(f"Starting forecast for {coin}")
        for future in range(2, 31):
            print("** n =", future)

            cf = CryptoForecast(
                epochs=args.epochs,
                batch_size=args.batch_size,
                ticker=coin,
                folds=args.folds,
                retrain=False,
                path=args.path,
                weights=args.weights,
                future_days=future,
                save=True,
                show=False,
                debug=False
            )

            cf.preprocess()
            cf.load_history()
            cf.predict_future()
            cf.postprocess()
            cf.visualize()
