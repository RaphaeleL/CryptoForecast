#!/usr/bin/env python3

import tensorflow as tf
import os
import warnings

from src.utils import print_help, parse_args
from src.forecast import CryptoForecast

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == "__main__":
    args, argparser = parse_args()

    cf = CryptoForecast(
        epochs=args.epochs,
        batch_size=args.batch_size,
        ticker=args.coin,
        folds=args.folds,
        retrain=args.retrain,
        path=args.path,
        weights=args.weights,
        future_days=args.future
    )

    if args.help:
        print_help(cf)

    cf.prepare()
    cf.load_history()
    cf.predict_future()
    cf.visualize()
