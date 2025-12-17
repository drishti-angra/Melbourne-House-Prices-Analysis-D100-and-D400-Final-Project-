from sklearn.metrics import r2_score
from pandas.api.types import is_numeric_dtype


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import dalex as dx




def evaluate_predictions(y_true_log, y_pred_log) -> pd.DataFrame:
    """
    Evaluate predictions for a log-price house price model.

    Reports metrics on:
    1) Log-price scale
    2) Original price scale

    Parameters
    ----------
    y_true_log : array-like
        True log house prices.
    y_pred_log : array-like
        Predicted log house prices.

    Returns
    -------
    pd.DataFrame
        metrics
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    # log price metrics
    log_true_mean = np.mean(y_true_log)
    log_pred_mean = np.mean(y_pred_log)

    log_mae = np.mean(np.abs(y_pred_log - y_true_log))
    log_rmse = np.sqrt(np.mean((y_pred_log - y_true_log) ** 2))
    log_bias = (log_pred_mean - log_true_mean) / log_true_mean
    log_mae_pct = log_mae / log_true_mean

    # price metrics
    price_true = np.exp(y_true_log)
    price_pred = np.exp(y_pred_log)

    price_true_mean = np.mean(price_true)
    price_pred_mean = np.mean(price_pred)

    price_mae = np.mean(np.abs(price_pred - price_true))
    price_rmse = np.sqrt(np.mean((price_pred - price_true) ** 2))
    price_bias = (price_pred_mean - price_true_mean) / price_true_mean
    price_mae_pct = price_mae / price_true_mean

    metrics = {

        "Log: True Mean": log_true_mean,
        "Log: Mean Prediction": log_pred_mean,
        "Log: MAE": log_mae,
        "Log: RMSE": log_rmse,
        "Log: Bias": log_bias,
        "Log: MAE / Mean": log_mae_pct,

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
    X_dalex = X.copy()

    # 1) due to "boolean value of NA Ambiguous error"
    X_dalex = X_dalex.apply(lambda s: s.astype(object).where(~pd.isna(s), np.nan))
    y_dalex = np.asarray(y, dtype=float)

    # 2) Convert numeric columns back to numeric for PDP
    numeric_cols = [
        "Distance", "Bedroom2", "Bathroom", "Car", "Landsize", "YearBuilt",
        "Lattitude", "Longtitude",
        "Year_binary",
        "Missing_Car", "Missing_BuildingArea", "Missing_YearBuilt",
        "Landsize_zero", "Bathroom_zero", "Bedroom2_zero",
    ]
    for col in numeric_cols:
        if col in X_dalex.columns:
            X_dalex[col] = pd.to_numeric(X_dalex[col], errors="coerce")

    

    explainer = dx.Explainer(model= model, data=X_dalex, y=y_dalex, label= label)

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



def plot_pdp(glm_explainer, lgbm_explainer, feature: str, variable_type: str):
    """
    Plotting PDP for a chosen feature

    Parameters
    ----------
    glm_explainer : dx.Explainer
        DALEX explainer for the GLM model.
    lgbm_explainer : dx.Explainer
        DALEX explainer for the LGBM model.
    feature : str
        Feature name to plot.
    variable_type : {"numerical", "categorical"}
        Explicit feature type for DALEX.
    """

    glm_pdp = glm_explainer.model_profile(variables=[feature], type="partial", variable_type =variable_type)
    lgbm_pdp = lgbm_explainer.model_profile(variables=[feature], type="partial", variable_type =variable_type)

    glm_pdp.plot(lgbm_pdp)






def lorenz_curve(y_true, y_pred):
    """
    Lorenz curve for regression without exposure weights.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    cumulated_samples : np.ndarray
        Cumulative share of samples (0..1)
    cumulated_true : np.ndarray
        Cumulative share of total y_true captured when sorting by y_pred
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ranking = np.argsort(y_pred)
    y_true_sorted = y_true[ranking]

    cum_true = np.cumsum(y_true_sorted)
    cum_true /= cum_true[-1] 

    cum_samples = np.linspace(0, 1, len(y_true_sorted))
    return cum_samples, cum_true




def plot_lorenz(y_true, glm_pred_log, lgbm_pred_log, xtitle):
    """
    Plot Lorenz curves for GLM and LGBM 

    Parameters
    ----------
    y_true_log : array-like
        True log(price).
    glm_pred_log : array-like
        Predicted log(price) from GLM.
    lgbm_pred_log : array-like
        Predicted log(price) from LGBM.
    xtitle: str
        title of the graph 

    """
    y_true = np.asarray(y_true, dtype=float)
    glm_pred_log = np.asarray(glm_pred_log, dtype=float)
    lgbm_pred_log = np.asarray(lgbm_pred_log, dtype=float)

    y_true = y_true
    glm_pred = glm_pred_log
    lgbm_pred = lgbm_pred_log
    title = xtitle

    # Compute curves
    x_glm, y_glm = lorenz_curve(y_true, glm_pred)
    x_lgbm, y_lgbm = lorenz_curve(y_true, lgbm_pred)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(x_glm, y_glm, label="GLM")
    plt.plot(x_lgbm, y_lgbm, label="LGBM")

    # Equality line
    plt.plot([0, 1], [0, 1], linestyle="--", label="Equality")

    plt.title(title)
    plt.xlabel("Cumulative share of properties- sorted by predicted price")
    plt.ylabel("Cumulative share of total true price")
    plt.legend()
    plt.show()