from sklearn.metrics import r2_score


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import dalex as dx




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



def plot_predicted_actual(ax, y_true, y_pred, title, xlabel, ylabel):
    """
    Scatter plot of predicted vs actual values with a y=x reference line.
    """
    ax.scatter(y_pred, y_true, alpha=0.25, s=12)

    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def dalex_explainer(model, X, y, label):
    """
    Creating Dalex explainer to find the most important features
    
    Parameters
    ----------
    model
        Fitted sklearn-compatible model or pipeline.
    X : pd.DataFrame
        Feature matrix used for explanation.
    y : array-like
        True target values.
    label : str
        Label used in DALEX plots (e.g. "GLM", "LGBM").

    Returns
    -------
    dx.Explainer
        DALEX explainer object.
    """

    explainer = dx.Explainer(model=model, data=X, y=y, label=label)
    return explainer

def feature_importance(explainer):
    """
    Using Dalex explainer to find the feature importance (using permutation importance)

    Parameters
    ----------
    explainer : dx.Explainer
        DALEX explainer object.

    Returns
    -------
    dalex.model_explanations.ModelParts
        Feature importance object.  

    """
    dalex_feature_importance = explainer.model_parts()

    return dalex_feature_importance