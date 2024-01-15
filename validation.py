import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

from utils import get_colored_text

keys = ["Monte Carlo Simulation", "Trend Analysis", "Asking a LLM"]
space = lambda x, y: "." * (len(max(keys, key=len)) - len(x) + 7 + y)
pss = lambda x, y, z, a: print(f"        {'└──' if a else '├──'} {x} {space(x, y)} {z}")  # Level 2
psa = lambda x, y, z, a: print(f"    {'└──' if a else '├──'} {x} {space(x, y)} {z}")  # Level 1
ps = lambda x, y, z: print(f"    {'└──' if z else '├──'} {x} {space(x, y)} ", end="")  # Level 1 without End


def convert_result_to_text(text):
    if text == None:
        return get_colored_text("n/a", "purple")
    elif text:
        return get_colored_text("passed", "green")
    elif not text:
        return get_colored_text("failed", "red")


def monte_carlo_simulation(cf, num_simulations=1000, num_days=30, confidence_interval=0.95, plot=False):
    ps(keys[0], 4, True)
    last_price = cf.data['Close'][-1]
    simulation_df = pd.DataFrame()
    # TODO: Thread this for loop
    for x in range(num_simulations):
        count = 0
        daily_volatility = cf.data['Close'].pct_change().std()
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
    print(convert_result_to_text(res))
    pss("Simulation Count", 0, f"#{num_simulations}", False)
    pss("Days", 0, num_days, False)
    pss("Confidence Interval", 0, f"{confidence_interval * 100}%", True)
    return res


def trend_analysis(cf):
    result, color = "n/a", "purple"
    ps(keys[1], 4, True)
    # TODO: Swing Trading also Stündliche Analyse
    print(convert_result_to_text(None))
    pss("Type", 0, "Swing Trading", False)
    pss("Trend Direction", 0, get_colored_text(result, color), True)


def ask_llm(cf):
    ps(keys[2], 4, True)
    # TODO: OpenAI ChatGPT API
    print(convert_result_to_text(None))


def validate(cf):
    print("└── Validation")

    monte_carlo_simulation(cf, 500, 365, 0.95, False)
    trend_analysis(cf)
    ask_llm(cf)
