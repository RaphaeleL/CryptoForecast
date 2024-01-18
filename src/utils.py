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
    diff_days = len(prediction) - len(cf.raw_data)
    end_date = cf.raw_data.index[-1] + pd.Timedelta(days=diff_days)
    date_range = pd.date_range(start=cf.raw_data.index[0], end=end_date, freq="D")
    prediction = pd.DataFrame(prediction, index=date_range, columns=["Prediction"])
    return prediction


def plot(cf):
    num_plots = 2
    _, axs = plt.subplots(num_plots, 1, figsize=(num_plots*10, num_plots*5))
    
    ax1 = axs[0]
    ax2 = axs[1]

    for ax in [ax1, ax2]:
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)

    ax1.plot(pred_only_for_plots(cf), label="Historical Prediction", color="red")
    ax1.plot(cf.forecast_data, label="Future Prediction", color="blue")
    ax1.plot(cf.data, label="Actual", color="green")
    ax1.set_title(f"{cf.ticker} History with Prediction")
    ax1.set_xlabel("Days")
    ax1.set_ylabel(f"Price in {cf.ticker.split('-')[1]}")
    ax1.legend()
    ax1.grid(True)
    ax1.set
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=150))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2.plot(cf.forecast_data[-cf.future_days*20:], label="Prediction")
    ax2.plot(cf.data[-cf.future_days:], label="Actual")
    ax2.set_title(f"{cf.ticker} Future Predictions for {cf.future_days} days")
    ax2.set_xlabel("Days")
    ax2.set_ylabel(f"Price in {cf.ticker.split('-')[1]}")
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=get_interval(cf.future_days)))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

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
