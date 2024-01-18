#!/usr/bin/env python3

import tensorflow as tf
import os
import time
import warnings

from src.utils import cprint, print_help
from src.forecast import CryptoForecast

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == "__main__":
    start_time = time.time()
    cf = CryptoForecast()
    if cf.args.help:
        print_help(cf)
    cf.load_history()
    cf.predict_future()
    diff = round(time.time() - start_time, 2)
    cprint(f"*** Needed {diff} seconds for {cf.ticker} with {cf.future_days} Days", "green" if diff < 60 else "red")
    cf.visualize()
