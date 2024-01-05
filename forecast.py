#!/usr/bin/env python3

from utils import load_history, predict_future, argument_parser, get_data, \
    get_weight_file_path, cprint, plot


def crypto_forecast(data, scaler, X, y, kf, args):
    trn, tst, prd, dur = [], [], [], args.prediction * 24
    model, trn, tst = load_history(args, kf, X, y, scaler, data, trn, tst)
    prd = predict_future(args, kf, X, y, scaler, data, prd, dur)
    return model, trn, tst, prd


if __name__ == "__main__":
    args = argument_parser()
    data, scaler, X, y, kf = get_data(args)
    forecast_result = crypto_forecast(data, scaler, X, y, kf, args)
    model, trn, tst, prd = forecast_result

    if args.retrain:
        model_filename = get_weight_file_path(args.coin)
        model.save_weights(model_filename)
        cprint(f"Saved model weights to '{model_filename}'", "green")

    if args.debug > 1:
        plot(args.coin, prd, args.prediction)
