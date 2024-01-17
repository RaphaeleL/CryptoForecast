#!/usr/bin/env python3

import tensorflow as tf
import os
import warnings

from src.forecast import CryptoForecast

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(cf=CryptoForecast()):

    cf.load_history()
    cf.predict_future(cf.future_days)
    cf.visualize()

if __name__ == "__main__":
    main()