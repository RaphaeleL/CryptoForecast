import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

keys = ["Monte Carlo Simulation", "Trend Analysis", "Asking a LLM"]


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
    cf, num_simulations=1000, num_days=30, confidence_interval=0.95, plot=False
):
    cf.metric += ps(keys[0], 4, True)
    last_price = cf.data["Close"][-1]
    simulation_df = pd.DataFrame()
    # TODO: Thread this for loop
    for x in range(num_simulations):
        count = 0
        daily_volatility = cf.data["Close"].pct_change().std()
        price_series = []
        price = last_price * (1 + np.random.normal(0, daily_volatility))
        price_series.append(price)
        # TODO: Thread this for loop
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


def trend_analysis(cf):
    result = "n/a"
    cf.metric += ps(keys[1], 4, True)
    # TODO: Swing Trading also Stündliche Analyse
    cf.metric += convert_result_to_text(None) + "\n"
    cf.metric += pss("Type", 0, "Swing Trading", False) + "\n"
    cf.metric += pss("Trend Direction", 0, result, True) + "\n"


def ask_llm(cf):
    cf.metric += ps(keys[2], 4, True)
    # TODO: OpenAI ChatGPT API
    cf.metric += convert_result_to_text(None) + "\n"


def validate(cf):
    cf.metric += "└── Validation\n"

    monte_carlo_simulation(cf, 500, 365, 0.95, False)
    trend_analysis(cf)
    ask_llm(cf)
