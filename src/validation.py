import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

keys = ["Monte Carlo Simulation", "Trend Analysis", "Asking a LLM", "Tweet Analysis", "Google Trend Analysis"]


def space(x, y):
    return "." * (len(max(keys, key=len)) - len(x) + 7 + y)


def pss(x, y, z, a):
    return f"        {'└──' if a else '├──'} {x} {space(x, y)} {z}"


def psa(x, y, z, a):
    return f"    {'└──' if a else '├──'} {x} {space(x, y)} {z}"


def ps(x, y, z):
    return f"    {'└──' if z else '├──'} {x} {space(x, y)} "


def convert_result_to_text(text):
    if text == None:
        return "n/a"
    elif text:
        return "passed"
    elif not text:
        return "failed"


def monte_carlo_simulation(
    cf, i, num_simulations=1000, num_days=30, confidence_interval=0.95, plot=False
):
    cf.metric += ps(keys[i], 4, True)
    last_price = cf.data["Close"][-1]
    simulation_df = pd.DataFrame()
    # TODO: Add threading
    for x in range(num_simulations):
        count = 0
        daily_volatility = cf.data["Close"].pct_change().std()
        price_series = []
        price = last_price * (1 + np.random.normal(0, daily_volatility))
        price_series.append(price)
        # TODO: Add threading
        for _ in range(num_days):
            if count == 251:
                break
            price = price_series[count] * (1 + np.random.normal(0, daily_volatility))
            price_series.append(price)
            count += 1
        simulation_df[x] = price_series

    lower_bound = simulation_df.quantile((1 - confidence_interval) / 2)
    upper_bound = simulation_df.quantile(1 - (1 - confidence_interval) / 2)
    # NOTE: This is the actual price of the stock or the last price of the Prediction
    # NOTE: Should we just use both? To double check?
    # actual_price = cf.forecast_data["Prediction"].iloc[0]
    actual_price = yf.Ticker(cf.ticker).history(period="1d")["Close"][0]

    if plot:
        plt.title(f"{cf.ticker} Monte Carlo Simulation")
        plt.plot(simulation_df)
        plt.show()

    res = (lower_bound <= actual_price).any() and (actual_price <= upper_bound).any()
    cf.metric += convert_result_to_text(res) + "\n"
    cf.metric += pss("Simulation Count", 0, f"#{num_simulations}", False) + "\n"
    cf.metric += pss("Days", 0, num_days, False) + "\n"
    cf.metric += pss("Confidence Interval", 0, f"{confidence_interval * 100}%", True) + "\n"
    return res


def ask_llm(cf, i):
    trends = {"ChatGPT": "n/a", "Google Bard": "n/a"} # TODO: Add more LLMs
    cf.metric += ps(keys[i], 4, True)
    # TODO: Implement a LLM Request
    result = all(value is True for value in trends.values())
    cf.metric += convert_result_to_text(result) + "\n"
    for index, llms in enumerate(trends.keys()):
        cf.metric += pss(llms, 0, trends[llms], True if index == len(trends) -1 else False) + "\n"


def tweet_analysis(cf, i):
    cf.metric += ps(keys[3], 4, True if i == len(keys) -1 else False)
    # TODO: Sentiment Analysis from Tweets
    cf.metric += convert_result_to_text(None) + "\n"
    

def google_trend_analysis(cf, i):
    cf.metric += ps(keys[i], 4, True if i == len(keys) -1 else False)
    # TODO: Google Trend Analysis
    #       https://github.com/bhushan23/Cryptocurrency-Analysis/blob/master/2_Google_Trends.ipynb
    cf.metric += convert_result_to_text(None) + "\n"

def validate(cf):
    cf.metric += "└── Validation\n"

    monte_carlo_simulation(cf, 0, 500, 365, 0.95, False)
    ask_llm(cf, 2)
    tweet_analysis(cf, 3)
    google_trend_analysis(cf, 4)