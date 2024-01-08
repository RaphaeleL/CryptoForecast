import math
from matplotlib import pyplot as plt, dates as mdates


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
        ax.legend()
        ax.grid(True)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot(prediction_days, forecast_data, ticker):
    pre_days = prediction_days * 12
    formatter = mdates.DateFormatter("%Y-%m-%d - %H:%M")
    x = forecast_data.head(pre_days).index
    y = forecast_data["Prediction"][:pre_days]
    plt.plot(x, y, label="Prediction", alpha=0.7)
    plt.title(f"{ticker} Future Predictions")
    plt.xlabel("Days")
    plt.ylabel(f"Price in {ticker.split('-')[1]}")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
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
