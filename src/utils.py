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


def get_interval(future_days):
    if future_days <= 7:
        return 1
    elif future_days <= 30:
        return 7
    elif future_days <= 90:
        return 30
    elif future_days <= 180:
        return 90


def pred_only_for_plots(cf):
    future_prediction = cf.model.predict(cf.X, verbose=0)
    prediction = cf.scaler.inverse_transform(future_prediction)

    start_date = cf.raw_data.index[0]
    if cf.args.min:
        total_minutes = len(prediction)
        date_range = pd.date_range(start=start_date, periods=total_minutes, freq='T')
    else:
        end_date = cf.raw_data.index[-1]
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    if len(date_range) != len(prediction):
        raise ValueError("Mismatch in the length of date range and predictions")

    prediction = pd.DataFrame(prediction, index=date_range, columns=["Prediction"])
    return prediction


def plot(cf):
    num_plots = 2
    fig, axs = plt.subplots(num_plots, 2, figsize=(num_plots*10, num_plots*5))

    time_format = "%Y-%m-%d %H:%M" if cf.args.min else "%Y-%m-%d"
    date_locator_interval = 1 if cf.args.min else get_interval(cf.future_days)

    plt.subplot(2, 2, 1)
    plt.plot(cf.forecast_data[-cf.future_days*20:], label="Prediction")
    plt.title(f"{cf.ticker} Future Predictions for {cf.future_days} {'min' if cf.args.min else 'days'}")
    plt.xlabel("Time")
    plt.ylabel(f"Price in {cf.ticker.split('-')[1]}")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(time_format))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=date_locator_interval))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")

    plt.subplot(2, 2, 2)
    plt.plot(cf.forecast_data[-cf.future_days*20:], label="Prediction")
    plt.plot(cf.data[-cf.future_days:], label="Actual")
    plt.title(f"{cf.ticker} History Data & Future Predictions for {cf.future_days} {'min' if cf.args.min else 'days'}")
    plt.xlabel("Time")
    plt.ylabel(f"Price in {cf.ticker.split('-')[1]}")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(time_format))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=date_locator_interval))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")

    plt.subplot(2, 1, 2)
    plt.plot(pred_only_for_plots(cf), label="Historical Prediction", color="red")
    plt.plot(cf.forecast_data, label="Future Prediction", color="blue")
    plt.plot(cf.data, label="Actual", color="green")
    plt.title(f"{cf.ticker} History with Prediction")
    plt.xlabel("Time")
    plt.ylabel(f"Price in {cf.ticker.split('-')[1]}")
    plt.legend()
    plt.grid(True)
    
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
