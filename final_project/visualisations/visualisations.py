from typing import Optional, List 

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from fitter import Fitter 
import scipy.stats as st



def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "Frequency",
) -> None:
    """
    Plot a histogram with a KDE (density curve) overlay for a numerical column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    column : str
        Name of the numerical column to plot.
    bins : int, optional
        Number of histogram bins. Default is 30.
    title : str, optional
        Title for the plot. Defaults to 'Histogram of <column>'.
    xlabel : str, optional
        Label for the x-axis. Defaults to the column name.
    ylabel : str, optional
        Label for the y-axis. Default is 'Frequency'.

    Returns
    -------
    None
        Displays the histogram with KDE overlay.
    """
    title = title or f"Histogram of {column}"
    xlabel = xlabel or column

    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    dist: Optional[List[str]] = None,
) -> None:
    """
    Plots a fit of the target variable against a list of distributions.

    Args:
        df (pd.DataFrame): pandas DataFrame
        column (str): column to analyse
        dist (list[str], optional): list of distributions to fit.
            If None, all distributions available in `fitter` are used.
    """
    f = Fitter(df[column].to_numpy(), distributions=dist)
    plt.figure(figsize=(10, 8))
    f.fit()
    plt.title(f"Distribution of {column}, best fit: {f.get_best()}")
    f.summary()
    plt.show()