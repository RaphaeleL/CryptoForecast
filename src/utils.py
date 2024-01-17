import os
import glob
import datetime
import pandas as pd
from matplotlib import pyplot as plt, dates as mdates


colors = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "purple": "\033[95m",
    "cyan": "\033[96m",
    "end": "\033[0m",
}


def plot(cf):
    num_plots = 2
    _, axs = plt.subplots(1, num_plots, figsize=(num_plots * 5, 5))

    axs[0].plot(cf.forecast_data, label="Prediction")
    axs[0].plot(cf.data, label="Actual")
    axs[0].set_title(f"{cf.ticker} Future Predictions")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel(f"Price in {cf.ticker.split('-')[1]}")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].xaxis_date()
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=120))
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right")

    axs[1].plot(cf.forecast_data[-cf.future_days*20:], label="Prediction")
    axs[1].plot(cf.data[-cf.future_days:], label="Actual")
    axs[1].set_title(f"{cf.ticker} Future Predictions")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel(f"Price in {cf.ticker.split('-')[1]}")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].xaxis_date()
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    filepath = create_cloud_path(cf.args.path, ticker=cf.ticker, typeof="plots", filetype="png")
    plt.savefig(filepath)
    plt.show()


def cprint(to_print, color, end="\n"):
    print(f"{colors.get(color, '')}{to_print}{colors['end']}", end=end)


def get_colored_text(to_print, color):
    return f"{colors.get(color, '')}{to_print}{colors['end']}"


def get_top_50_tickers():
    return list(pd.read_csv("data/top_50_names.csv")["Name"])


def get_weight_file_tickers():
    return glob.glob("weights/*.h5")


def get_full_ticker_list():
    result = []
    for weight_ticker in get_weight_file_tickers():
        for ticker in get_top_50_tickers():
            if ticker in weight_ticker:
                result.append(ticker + "-EUR")
    return result


def extract_min_max(cf):
    current_fmt, new_fmt = (
        "%Y-%m-%d %H:%M:%S" if not cf.args.minutely else "%Y-%m-%d %H:%M:%S%z",
        "%d. %b %Y - %H:%M",
    )

    max_difference = 0
    min_index, max_index = None, None
    min_value, max_value = None, None
    global_min_value, global_max_value = float("inf"), float("-inf")
    global_min_index, global_max_index = None, None

    for i in range(len(cf.forecast_data) - 1):
        for j in range(i + 1, len(cf.forecast_data)):
            difference = (
                cf.forecast_data["Prediction"].iloc[j]
                - cf.forecast_data["Prediction"].iloc[i]
            )
            if difference > max_difference:
                max_difference = difference
                min_index = cf.forecast_data.index[i]
                max_index = cf.forecast_data.index[j]
                min_value = cf.forecast_data["Prediction"].iloc[i]
                max_value = cf.forecast_data["Prediction"].iloc[j]

        current_value = cf.forecast_data["Prediction"].iloc[i]
        if current_value < global_min_value:
            global_min_value = current_value
            global_min_index = cf.forecast_data.index[i]
        if current_value > global_max_value:
            global_max_value = current_value
            global_max_index = cf.forecast_data.index[i]

    min_index = datetime.datetime.strptime(str(min_index), current_fmt).strftime(new_fmt)
    max_index = datetime.datetime.strptime(str(max_index), current_fmt).strftime(new_fmt)
    global_min_index = datetime.datetime.strptime(
        str(global_min_index), current_fmt).strftime(new_fmt)
    global_max_index = datetime.datetime.strptime(
        str(global_max_index), current_fmt).strftime(new_fmt)

    return (
        min_index,
        min_value,
        max_index,
        max_value,
        global_min_index,
        global_min_value,
        global_max_index,
        global_max_value,
    )


def save_prediction(predictions, path):
    predictions.to_csv(path, index=True)

def get_dafault_bw_path():
    return os.path.join(os.path.expanduser('~'), "bwSyncShare", "PadWise-Trading")


def create_cloud_path(cloudpath, ticker, typeof, filetype):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    path = os.path.join(
        cloudpath,
        typeof,
        ticker
    )
    filename = f"{timestamp}.{filetype}"

    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)

    return filepath
