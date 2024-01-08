#!/usr/bin/env python3

from forecast import CryptoForecast


if __name__ == "__main__":

    # TODO:
    #   - VALIDATION
    #       - Trend Analysis (Up, Down, Sideways, etc.)
    #       - LLMs (ChatGPT, LLaMa, etc.))
    #       - For Step 3

    cf = CryptoForecast()

    if cf.should_retrain:
        cf.load_history()
        exit()

    if cf.args.agents:
        results = cf.use_agents()
        cf.visualize_agents(results)
    else:
        cf.load_history()
        cf.predict_future()
        cf.visualize()
