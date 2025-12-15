from sklearn.metrics import r2_score


import numpy as np 
import pandas as pd


def evaluate_predictions(y_true, y_pred) -> pd.DataFrame:
    """
    Evaluate predictions for the GLM and LGBM models.

    Metrics outputted are Mean true log(prices), mean predictions, RMSE, MAE, R^2, bias, and MAE as a percentage of mean log(prices)
    of the mean house price.

    Parameters
    ----------
    y_true
        True target values (log house prices).
    y_pred
        Predicted target values.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics (metrics as rows).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))

    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    evals = {
        "Mean True log(Price)": mean_true,
        "Mean Prediction of log(Prices)": mean_pred,
        "Bias": (mean_pred - mean_true) / mean_true,
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r2,
        "MAE as \%\ of mean log(prices) ": mae / mean_true,
    }

    return pd.DataFrame(evals, index=[0]).T



import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def evaluate_predictions_full(
    y_true_log,
    y_pred_log,
) -> pd.DataFrame:
    """
    Evaluate predictions for log-price house price models.

    Reports metrics on:
    - log-price scale
    - original price scale (via exp transformation)

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

    # -----------------
    # Log-scale metrics
    # -----------------
    mean_true_log = np.mean(y_true_log)
    mean_pred_log = np.mean(y_pred_log)

    mae_log = np.mean(np.abs(y_pred_log - y_true_log))
    rmse_log = np.sqrt(np.mean((y_pred_log - y_true_log) ** 2))
    r2_log = r2_score(y_true_log, y_pred_log)

    # -------------------
    # Price-scale metrics
    # -------------------
    price_true = np.exp(y_true_log)
    price_pred = np.exp(y_pred_log)

    mean_true_price = np.mean(price_true)
    mean_pred_price = np.mean(price_pred)

    mae_price = np.mean(np.abs(price_pred - price_true))
    rmse_price = np.sqrt(np.mean((price_pred - price_true) ** 2))

    mae_price_pct = mae_price / mean_true_price

    # multiplicative percentage error (very interpretable)
    mean_abs_pct_error = np.mean(np.abs(price_pred / price_true - 1))

    evals = {
        # log scale
        "Mean True (log)": mean_true_log,
        "Mean Pred (log)": mean_pred_log,
        "MAE (log)": mae_log,
        "RMSE (log)": rmse_log,
        "R^2 (log)": r2_log,

        # price scale
        "Mean True Price": mean_true_price,
        "Mean Pred Price": mean_pred_price,
        "MAE Price": mae_price,
        "RMSE Price": rmse_price,
        "MAE / Mean Price": mae_price_pct,
        "Mean Abs % Error": mean_abs_pct_error,
    }

    return pd.DataFrame(evals, index=[0]).T
