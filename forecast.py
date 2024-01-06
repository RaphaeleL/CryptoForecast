#!/usr/bin/env python3

from utils import load_history, predict_future, cprint, \
    preprocess, postprocess, get_weight_file_path


if __name__ == "__main__":
    args, data, scaler, X, y, = preprocess()

    if args.retrain:
        model, trn, tst = load_history(args, X, y, scaler, data)
        model_filename = get_weight_file_path(args.coin)
        model.save_weights(model_filename)
        cprint(f"Saved model weights to '{model_filename}'", "green")
    else:
        dur = args.prediction * 24
        model, trn, tst = load_history(args, X, y, scaler, data)
        prd = predict_future(args, X, y, scaler, data, dur)
        postprocess(args, prd)
