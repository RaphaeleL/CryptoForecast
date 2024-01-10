#!/usr/bin/env python3

from forecast import CryptoForecast


def main(cf):

    if cf.should_retrain:
        cf.load_history()
        cf.stop_time("to retrain the model")
        exit()

    if cf.args.agents:
        results = cf.use_agents()
        cf.stop_time("to predict the future with Agents.")
        cf.visualize_agents(results)
    else:
        cf.load_history()
        cf.predict_future()
        cf.stop_time("to predict the future with Weights.")
        cf.visualize()


if __name__ == "__main__":
    main(CryptoForecast())
