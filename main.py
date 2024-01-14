#!/usr/bin/env python3

import time
import tensorflow as tf
import os
import warnings

from forecast import CryptoForecast
from utils import get_full_ticker_list, get_colored_text
from validation import validate

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(cf=CryptoForecast()):

    if cf.should_retrain:
        cf.load_history()
        cf.stop_time()
        cf.show_result()
        return

    if cf.args.debug:
        actual_data = cf.backtest()
        cf.stop_time()
        cf.show_result()
        cf.visualize_backtest(actual_data)
        return

    if cf.args.auto:
        for ticker in get_full_ticker_list():
            tcf = CryptoForecast(ticker)
            tcf.load_history()
            tcf.predict_future()
            tcf.stop_time()
            tcf.show_result()
            validate(cf)
        return

    cf.load_history()
    cf.predict_future()
    cf.stop_time()
    cf.show_result()
    validate(cf)
    cf.visualize()

    # TODO: Save the Predicted Data to a CSV file
    # TODO: Save the Metrics to a TXT File


if __name__ == "__main__":
    main()