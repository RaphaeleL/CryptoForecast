
from matplotlib import pyplot as plt, dates as mdates


def plot(cf, data):
    pre_days = cf.prediction_days * 12
    formatter = mdates.DateFormatter("%Y-%m-%d - %H:%M")
    x = data.head(pre_days).index
    y = data["Prediction"][:pre_days]
    plt.plot(x, y, label="Prediction", alpha=0.7)
    plt.title(f"{cf.ticker} Future Predictions")
    plt.xlabel("Days")
    plt.ylabel(f"Price in {cf.ticker.split('-')[1]}")
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
