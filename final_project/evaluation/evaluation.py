from sklearn.metrics import r2_score


import numpy as np 
import pandas as pd


import numpy as np
import pandas as pd


def evaluate_predictions(y_true_log, y_pred_log) -> pd.DataFrame:
    """
    Evaluate predictions for a log-price house price model.

    Reports metrics on:
    1) Log-price scale
    2) Original price scale (via exp transformation)

    Parameters
    ----------
    y_true_log : array-like
        True log house prices.
    y_pred_log : array-like
        Predicted log house prices.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics (metrics as rows).
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    # =====================
    # Log-price metrics
    # =====================
    log_true_mean = np.mean(y_true_log)
    log_pred_mean = np.mean(y_pred_log)

    log_mae = np.mean(np.abs(y_pred_log - y_true_log))
    log_rmse = np.sqrt(np.mean((y_pred_log - y_true_log) ** 2))
    log_bias = (log_pred_mean - log_true_mean) / log_true_mean
    log_mae_pct = log_mae / log_true_mean

    # =====================
    # Price-scale metrics
    # =====================
    price_true = np.exp(y_true_log)
    price_pred = np.exp(y_pred_log)

    price_true_mean = np.mean(price_true)
    price_pred_mean = np.mean(price_pred)

    price_mae = np.mean(np.abs(price_pred - price_true))
    price_rmse = np.sqrt(np.mean((price_pred - price_true) ** 2))
    price_bias = (price_pred_mean - price_true_mean) / price_true_mean
    price_mae_pct = price_mae / price_true_mean

    metrics = {
        # log-price
        "Log: True Mean": log_true_mean,
        "Log: Mean Prediction": log_pred_mean,
        "Log: MAE": log_mae,
        "Log: RMSE": log_rmse,
        "Log: Bias": log_bias,
        "Log: MAE / Mean": log_mae_pct,

        # price
        "Price: True Mean": price_true_mean,
        "Price: Mean Prediction": price_pred_mean,
        "Price: MAE": price_mae,
        "Price: RMSE": price_rmse,
        "Price: Bias": price_bias,
        "Price: MAE / Mean": price_mae_pct,
    }

    return pd.DataFrame(metrics, index=[0]).T


