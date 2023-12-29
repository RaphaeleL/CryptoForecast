import threading, os, warnings
from utils import *
import pretty_errors

warnings.filterwarnings('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  

def crypto_forecast(data, scaler, X, y, kf, agent, args, test_pred, test_actu, real_pred, models, prediction_days=1):
    """Forecast cryptocurrency prices."""
    model, test_pred, test_actu = load_history(args, kf, X, y, agent, scaler, data, test_pred, test_actu)
    real_pred = predict_future(args, kf, X, y, agent, scaler, data, real_pred, prediction_days * 24)
    models[agent] = model

def main(coin, data, scaler, X, y, kf, args):
    threads, test_pred, test_actu, real_pred = [], [], [], []
    models = {}

    for agent in range(args.agents):
        thread = threading.Thread(
            target=crypto_forecast,
            args=(data, scaler, X, y, kf, agent, args, test_pred, test_actu, real_pred, models, args.prediction)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    performance_data = evaluate_agent_performance(test_actu, test_pred)
    best_agent = select_best_agent(performance_data)
    performance_output(args, real_pred, best_agent, coin)

    if args.save:
        model_filename = get_weight_file_path(coin)
        models[best_agent].save_weights(model_filename)
        print_colored(f"Saved best model weights of Agent {best_agent+1} to '{model_filename}'", "green")

    if args.debug > 2: plot(coin, agent, test_pred, test_actu, real_pred, args.prediction)

if __name__ == "__main__":
    args = argument_parser()
    data, scaler, X, y, kf = get_data(args)
    main(args.coin, data, scaler, X, y, kf, args)