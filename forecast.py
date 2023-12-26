import threading
import pandas as pd
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from utils import (
    evaluate_agent_performance,
    select_best_agent,
    need_retraining,
    load_and_preprocess_data,
    normalize_data,
    create_dataset,
    plot, 
    plot_all,
    split, 
    train,
    argument_parser,
    check,
    print_colored,
    print_data
)

def map_predictions_to_dates(start_date, predictions):
    """Map predictions to dates."""
    dates = pd.date_range(start_date, periods=len(predictions), freq='H')
    date_predictions = pd.DataFrame(predictions, index=dates, columns=['Prediction'])
    return date_predictions

def crypto_forecast(data, scaler, X, y, kf, agent, args, test_pred, test_actu, real_pred, prediction_days=1):
    """Forecast cryptocurrency prices."""
    predictions = []
    actuals = []
    real_predictions = []

    for index, (train_index, test_index) in enumerate(kf.split(X)):
        print_colored(f"Training Agent {agent+1}/{args.agents} for Fold {index+1}/{args.folds}", "green")
        X_train, X_test, y_train, y_test = split(X, y, train_index, test_index)
        model = train(X, y, X_train, y_train, args.batch_size, args.epochs)
        prediction = scaler.inverse_transform(model.predict(X_test))
        predictions.extend(prediction)
        actuals.extend(scaler.inverse_transform(y_test.reshape(-1, 1)))

    test_dates = data.index[test_index].to_pydatetime()
    test_pred.append(pd.DataFrame(prediction, index=test_dates, columns=['Prediction']))
    test_actu.append(pd.DataFrame(scaler.inverse_transform(y_test.reshape(-1, 1)), index=test_dates, columns=['Actual']))

    for index, (train_index, _) in enumerate(kf.split(X)):
        print_colored(f"Training Agent {agent+1}/{args.agents} for Fold {index+1}/{args.folds}", "green")
        X_train, y_train = split(X, y, train_index) 
        model = train(X, y, X_train, y_train, args.batch_size, args.epochs)
        prediction = scaler.inverse_transform(model.predict(X[-(prediction_days*12):]))
        real_predictions.extend(prediction)

    starting_index = len(test_pred[0]) + 1
    last_day = test_actu[-1].index[-1]
    next_day = last_day + pd.Timedelta(hours=1)
    future_dates = pd.date_range(start=next_day, periods=len(real_predictions), freq='H')
    real_pred.append(pd.DataFrame(real_predictions, index=future_dates, columns=['Prediction']))

    if not check(test_actu[-1].iloc[-1].item(), real_pred[0].iloc[starting_index].item()):
        border = "*****************************************************"
        print_colored(border, "red")
        print_colored(f"ERROR (Agent {agent+1})", "red")
        print_colored("Prediction is not accurate enough.", "red")
        print_colored(f"  > Actual Value:    {round(test_actu[-1][-1][0])}", "green")
        print_colored(f"  > Predicted Value: {round(real_pred[0][starting_index][0])}", "blue")
        print_colored(border, "red")

    return test_pred, test_actu, real_pred

def main(data, scaler, X, y, kf, args):
    # TODO: print a list of agents and folds with done, in progress and not started
    threads, test_pred, test_actu, real_pred = [], [], [], []

    for agent in range(args.agents):
        thread = threading.Thread(
            target=crypto_forecast,
            args=(data, scaler, X, y, kf, agent, args, test_pred, test_actu, real_pred, args.prediction),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    performance_data = evaluate_agent_performance(test_actu, test_pred)
    best_agent = select_best_agent(performance_data)

    if need_retraining(best_agent, performance_data):
        print_colored(f"Need to Train the Model again...", "red")
        print_colored(f"  > Best Agent: {best_agent+1}", "red")
        print_colored(f"  > Performance: {round(performance_data[best_agent], 2)}", "red")
        # main(data, scaler, X, y, kf, args)

    # print_data(real_pred)
    if args.show_all:
        plot_all(args.coin, best_agent, real_pred)
    plot(args.coin, best_agent, test_pred[best_agent], test_actu[best_agent], real_pred[best_agent])

if __name__ == "__main__":
    args = argument_parser()
    data = load_and_preprocess_data(args.coin)
    scaler, normalized_data = normalize_data(data)
    X, y = create_dataset(normalized_data)
    kf = KFold(n_splits=args.folds, shuffle=False)

    main(data, scaler, X, y, kf, args)