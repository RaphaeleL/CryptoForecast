import os
import glob
import datetime
import argparse
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
    fig, axs = plt.subplots(num_plots, 2, figsize=(num_plots*10, num_plots*5))

    plt.subplot(2, 2, 1)
    plt.plot(cf.forecast_data[-cf.future_days*20:], label="Prediction")
    plt.title(f"{cf.ticker} Future Predictions for {cf.future_days} days")
    plt.ylabel(f"Price in {cf.ticker.split('-')[1] if '-' in cf.ticker else ''}")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=get_interval(cf.future_days)))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")

    plt.subplot(2, 2, 2)
    plt.plot(cf.forecast_data[-cf.future_days*20:], label="Prediction")
    plt.plot(cf.data[-cf.future_days:], label="Actual")
    plt.title(f"{cf.ticker} History Data & Future Predictions for {cf.future_days} days")
    plt.ylabel(f"Price in {cf.ticker.split('-')[1] if '-' in cf.ticker else ''}")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=get_interval(cf.future_days)))
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")

    plt.subplot(2, 1, 2)
    plt.plot(pred_only_for_plots(cf), label="Historical Prediction", color="red")
    plt.plot(cf.forecast_data, label="Future Prediction", color="blue")
    plt.plot(cf.data, label="Actual", color="green")
    plt.title(f"{cf.ticker} History with Prediction")
    plt.ylabel(f"Price in {cf.ticker.split('-')[1] if '-' in cf.ticker else ''}")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    filepath = create_cloud_path(cf.path, ticker=cf.ticker, typeof="plots", filetype="png")
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


def print_help(cf):
    print("PadWise-Trading", end="\n\n")
    print("Usage: python3 main.py [options]", end="\n\n")
    print("Options:", end="\n\n")
    
    for action in cf.argparser._actions:
        arg_names = ', '.join(action.option_strings)
        default = action.default
        help_text = action.help
        type_str = str(action.type) if action.type is not None else "None"

        if not isinstance(default, (str, list, tuple)):
            default = str(default)
        else:
            default = ', '.join(map(str, default)) if isinstance(default, (list, tuple)) else default
            
        arg_line = f"  {arg_names :<20} {help_text}"
        default_line = f"\t\t\t  Default: {default}"
        type_line = f"\t\t\t  Type:    {type_str}"

        print(arg_line)
        
        if type_str != "None":
            print(default_line)
            print(type_line)
        print()

    exit()


def parse_args():
    argparser = argparse.ArgumentParser(add_help=False)
    argparser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
    argparser.add_argument("-c", "--coin", type=str, default="LTC-EUR", help="Coin to predict")
    argparser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size")
    argparser.add_argument("-e", "--epochs", type=int, default=200, help="Number of epochs")
    argparser.add_argument("-f", "--folds", type=int, default=6, help="Number of folds")
    argparser.add_argument("-t", "--retrain", action="store_true", help="(Re-)train the model")
    argparser.add_argument("-p", "--path", type=str, default=get_dafault_bw_path(), help="Path to save the results")
    argparser.add_argument("-w", "--weights", type=str, default=None, help="Path to model weight file")
    argparser.add_argument("-d", "--future", type=int, default=7, help="Number of days to predict")
    args = argparser.parse_args()
    return args, argparser