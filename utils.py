import math
from matplotlib import pyplot as plt, dates as mdates


formatter = mdates.DateFormatter("%Y-%m-%d %H:%M")
locator = mdates.DayLocator(interval=1)


def style_plot(ax):
    ax.legend()
    ax.grid(True)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def plot_multiple(cfs):
    num_agents = len(cfs)
    rows = math.ceil(math.sqrt(num_agents))
    cols = math.ceil(num_agents / rows)
    plt.figure(figsize=(15, rows * 5))
    for i, cf in enumerate(cfs):
        pre_days = cf.prediction_days * 12
        ax = plt.subplot(rows, cols, i + 1)
        x = cf.forecast_data.head(pre_days).index
        y = cf.forecast_data["Prediction"][:pre_days]
        plt.plot(x, y, label="Prediction", alpha=0.7)
        ax.set_title(f"{cf.ticker} Price Prediction by Agent {i+1}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price in $")
        style_plot(ax)
    plt.tight_layout()
    plt.show()


def plot(prediction_days, forecast_data, ticker):
    pre_days = prediction_days * 12
    x = forecast_data.head(pre_days).index
    y = forecast_data["Prediction"][:pre_days]
    plt.plot(x, y, label="Prediction", alpha=0.7)
    plt.title(f"{ticker} Future Predictions")
    plt.xlabel("Days")
    plt.ylabel(f"Price in {ticker.split('-')[1]}")
    style_plot(plt)
    plt.tight_layout()
    plt.show()


def cprint(to_print, color, end="\n"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "end": "\033[0m",
    }
    print(f"{colors.get(color, '')}{to_print}{colors['end']}", end=end)


def plot_backtest(forecast_data, actual_data, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast_data.index, forecast_data["Prediction"], label="Predicted Data", alpha=0.7)
    ax.plot(actual_data.index, actual_data["Close"], label="Actual Data", alpha=0.7)
    ax.set_title(f"{ticker} Backtest Results")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    style_plot(ax)
    ax.get_xaxis().set_visible(False)
    plt.show()
