import argparse
import tensorflow as tf
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras import Input, Model
from keras.layers import Dense, Conv1D, Dropout, LayerNormalization, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.optimizers.legacy import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from utils import load_and_preprocess_data, normalize_data, create_dataset, plot_predictions

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, epsilon=1e-6, attention_axes=None, kernel_size=1):
    """
    Creates a single transformer block.
    """
    x = LayerNormalization(epsilon=epsilon)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout,
        attention_axes=attention_axes
        )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=epsilon)(res)
    x = Conv1D(filters=ff_dim, kernel_size=kernel_size, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=kernel_size)(x)
    return x + res

def build_transformer(num_features, head_size=128, num_heads=8, ff_dim=4, num_trans_blocks=6, mlp_units=[512], dropout=0.10, mlp_dropout=0.10, attention_axes=1, epsilon=1e-6, kernel_size=1):
    """
    Creates final model by building many transformer blocks.
    """
    inputs = Input(shape=(1, num_features))
    x = inputs
    for _ in range(num_trans_blocks):
        x = transformer_encoder(x, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, attention_axes=attention_axes, kernel_size=kernel_size, epsilon=epsilon)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)

    outputs = Dense(1)(x)
    return Model(inputs, outputs)

def fit_improved_transformer(model, X_train, y_train, X_val, y_val):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mape'])
    lr_reducer = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    return history

def fit_transformer(transformer, X_train, y_train, X_test, y_test):
    """
    Compiles, fits, and evaluates our transformer.
    """
    transformer.compile(
        loss="mse",
        optimizer=Adam(learning_rate=1e-3),
        metrics=["mae", 'mape'])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
    start = time.time()
    hist = transformer.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=200, verbose=1, callbacks=callbacks)
    print(time.time() - start)
    return hist, transformer

def cross_validate_transformer(X, y, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_number = 0
    all_fold_predictions = []
    all_fold_actuals = []
    all_fold_metrics = []

    for train_index, val_index in kf.split(X):
        fold_number += 1
        print(f"Training on fold {fold_number}...")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        transformer = build_transformer(X.shape[2])
        history = fit_improved_transformer(transformer, X_train, y_train, X_val, y_val)
        valPredict = scaler.inverse_transform(transformer.predict(X_val))
        valY = scaler.inverse_transform(y_val.reshape(-1, 1))
        all_fold_predictions.append(valPredict)
        all_fold_actuals.append(valY)
        fold_metrics = {
            'mae': np.mean(history.history['val_mae']),
            'mse': np.mean(history.history['val_loss']),
            'mape': np.mean(history.history['val_mape'])
        }
        all_fold_metrics.append(fold_metrics)
        print(f"Fold {fold_number} MAE: {fold_metrics['mae']}, MSE: {fold_metrics['mse']}, MAPE: {fold_metrics['mape']}")

    return all_fold_predictions, all_fold_actuals, all_fold_metrics

def main(coin, batch_size, epochs, dataset_path, dataset_name, real_pred, reverse_plot):
    if real_pred == -1:
        data = load_and_preprocess_data(dataset_path)
        scaler, normalized_data = normalize_data(data)
        X, y = create_dataset(normalized_data)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        improved_transformer = build_transformer(X_train.shape[2])
        history = fit_improved_transformer(improved_transformer, X_train, y_train, X_val, y_val)
        valPredict = scaler.inverse_transform(improved_transformer.predict(X_val))
        valY = scaler.inverse_transform(y_val.reshape(-1, 1))
        plot_predictions(valPredict, coin, reverse_plot, dataset_name, testY=valY)
    else: 
        data = load_and_preprocess_data(dataset_path)
        scaler, normalized_data = normalize_data(data)
        X, y = create_dataset(normalized_data)
        transformer = build_transformer(X.shape[2])
        hist, transformer = fit_transformer(transformer, X, y, X, y)
        testPredict = scaler.inverse_transform(transformer.predict(X))
        plot_predictions(testPredict[-real_pred:], coin, reverse_plot, dataset_name)
