#!/usr/bin/env python3

import tensorflow as tf
import os
import warnings

from src.forecast import CryptoForecast
from src.utils import get_full_ticker_list

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(cf=CryptoForecast()):

    if cf.should_retrain:
        cf.load_history()
        cf.stop_time()
        return

    if cf.args.debug:
        actual_data = cf.backtest()
        cf.stop_time()
        cf.visualize_backtest(actual_data)
        return

    if cf.args.auto:
        for ticker in get_full_ticker_list(): # TODO: Add threading
            tcf = CryptoForecast(ticker)
            tcf.load_history()
            tcf.predict_future()
            tcf.stop_time()
            tcf.generate_metric()
        return

    cf.load_history()
    cf.predict_future()
    cf.stop_time()
    cf.generate_metric()
    cf.visualize()

if __name__ == "__main__":
    main()