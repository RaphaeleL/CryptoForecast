from matplotlib import pyplot as plt, dates as mdates

import datetime


colors = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "purple": "\033[95m",
    "cyan": "\033[96m",
    "end": "\033[0m",
}
formatter = mdates.DateFormatter("%Y-%m-%d %H:%M")
locator = mdates.DayLocator(interval=1)


def style_plot(ax):
    ax.legend()
    ax.grid(True)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def plot(forecast_data, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = forecast_data.index
    y = forecast_data["Prediction"]
    ax.plot(x, y, label="Prediction", alpha=0.7)
    ax.set_title(f"{ticker} Future Predictions")
    ax.set_xlabel("Days")
    ax.set_ylabel(f"Price in {ticker.split('-')[1]}")
    style_plot(ax)
    plt.tight_layout()
    plt.show()


def plot_backtest(forecast_data, actual_data, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast_data.index, forecast_data["Prediction"], label="Predicted Data", alpha=0.7)
    ax.plot(actual_data.index, actual_data["Close"], label="Actual Data", alpha=0.7)
    ax.set_title(f"{ticker} Backtest Results")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    style_plot(ax)
    ax.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def cprint(to_print, color, end="\n"):
    print(f"{colors.get(color, '')}{to_print}{colors['end']}", end=end)


def get_colored_text(to_print, color):
    return f"{colors.get(color, '')}{to_print}{colors['end']}"


def get_full_ticker_list():
    import glob
    result = []
    for ticker in glob.glob("weights/*.h5"):
        result.append(ticker.split("/")[1].split(".")[0])
    return result


def extract_min_max(cf):
    current_fmt, new_fmt = "%Y-%m-%d %H:%M:%S" if not cf.args.minutely else "%Y-%m-%d %H:%M:%S%z", "%d. %b %Y - %H:%M"

    max_difference = 0
    min_index, max_index = None, None
    min_value, max_value = None, None

    for i in range(len(cf.forecast_data) - 1):
        for j in range(i+1, len(cf.forecast_data)):
            difference = cf.forecast_data['Prediction'].iloc[j] - cf.forecast_data['Prediction'].iloc[i]
            if difference > max_difference:
                max_difference = difference
                min_index = cf.forecast_data.index[i]
                max_index = cf.forecast_data.index[j]
                min_value = cf.forecast_data["Prediction"].iloc[i]
                max_value = cf.forecast_data["Prediction"].iloc[j]
    min_index = datetime.datetime.strptime(str(min_index), current_fmt).strftime(new_fmt)
    max_index = datetime.datetime.strptime(str(max_index), current_fmt).strftime(new_fmt)
    min_str = f"{min_value:.2f} {cf.ticker.split('-')[1]}"
    max_str = f"{max_value:.2f} {cf.ticker.split('-')[1]}"
    return min_str, min_index, min_value, max_str, max_index, max_value
