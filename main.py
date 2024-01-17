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

    cf.load_history()
    cf.predict_future(cf.future_days)
    # cf.stop_time()
    # cf.generate_metric()
    cf.visualize()

if __name__ == "__main__":
    main()