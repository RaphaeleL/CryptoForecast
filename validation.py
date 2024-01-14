from utils import get_colored_text

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

keys = ["Monte Carlo Simulation", "Trend Analysis", "Asking a LLM"]
space = lambda x, y: "." * (len(max(keys, key=len)) - len(x) + 5 + y)


def convert_result_to_text(text):
    if text:
        return get_colored_text("passed", "green")
    elif not text:
        return get_colored_text("failed", "red")
    else:
        return get_colored_text("n/a", "yellow")


def monte_carlo_simulation(cf, num_simulations=1000, num_days=30, confidence_interval=0.95, plot=False):
    last_price = cf.data['Close'][-1]
    simulation_df = pd.DataFrame()
    for x in range(num_simulations):
        count = 0
        daily_volatility = cf.data['Close'].pct_change().std()
        price_series = []
        price = last_price * (1 + np.random.normal(0, daily_volatility))
        price_series.append(price)
        for y in range(num_days):
            if count == 251:
                break
            price = price_series[count] * (1 + np.random.normal(0, daily_volatility))
            price_series.append(price)
            count += 1
        simulation_df[x] = price_series

    lower_bound = simulation_df.quantile((1 - confidence_interval) / 2)
    upper_bound = simulation_df.quantile(1 - (1 - confidence_interval) / 2)
    actual_price = yf.Ticker(cf.ticker).history(period="1d")["Close"][0]

    if plot:
        plt.title(f"{cf.ticker} Monte Carlo Simulation")
        plt.plot(simulation_df)
        plt.show()

    return (lower_bound <= actual_price).any() and (actual_price <= upper_bound).any()


def trend_analysis(cf):
    # TODO: Swing Trading also Stündliche Analyse
    return False


def llm(cf):
    # TODO: OpenAI ChatGPT API
    return False


def validate(cf):
    results = {
            "Monte Carlo Simulation": monte_carlo_simulation(cf),
            "Trend Analysis": trend_analysis(cf),
            "Asking a LLM": llm(cf)
    }

    for index, key in enumerate(keys):
        delim = "└──" if index == len(keys) - 1 else "├──"
        print(f"{delim} {key} {space(key, 0)} {convert_result_to_text(results[key])}")
