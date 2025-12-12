from typing import Optional, List 

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from fitter import Fitter 
import scipy.stats as st
import seaborn as sns



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
    sns.histplot(df[column].dropna(), bins=bins, kde=True)
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



def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    ylabel: str | None = None,
) -> None:
    """
    Create a boxplot for a given column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to plot.
        ylabel (str, optional): Custom y-axis label. Defaults to the column name.
    """
    plt.figure(figsize=(6, 5))

    plt.boxplot(df[column].dropna(), vert=True)

    label_to_use = ylabel if ylabel is not None else column
    plt.ylabel(label_to_use)

    plt.xlabel("")

    plt.title(f"Box Plot of {column}")
    plt.show()


def plot_grouped_boxplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot boxplots of `y_col` for each category of `x_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x_col : str
        Column to use on the x-axis (groups/categories).
    y_col : str
        Column to use on the y-axis (values for boxplots).
    xlabel : str, optional
        Custom x-axis label. Defaults to `x_col`.
    ylabel : str, optional
        Custom y-axis label. Defaults to `y_col`.
    title : str, optional
        Plot title. Defaults to 'Box Plot of <y_col> by <x_col>'.
    """
    plt.figure(figsize=(12, 5))

    sns.boxplot(x=x_col, y=y_col, data=df)

    plt.xlabel(xlabel if xlabel is not None else x_col)
    plt.ylabel(ylabel if ylabel is not None else y_col)
    plt.title(title if title is not None else f"Box Plot of {y_col} by {x_col}")

    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_value_counts(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
    ylabel: str | None = "Count",
) -> None:
    """
    Plot a bar chart of value counts for a categorical variable.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to plot.
        title (str, optional): Chart title. Defaults to 
            "Value Counts of <column>".
        ylabel (str, optional): y-axis label. Defaults to "Count".
    """
    counts = df[column].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")

    plot_title = title if title is not None else f"Value Counts of {column}"
    plt.title(plot_title)
    plt.ylabel(ylabel)
    plt.xlabel(column)

    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_kde_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "Density",
    title: Optional[str] = None,
) -> None:
    """
    Plot kernel density estimates of a numeric column, grouped by another column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value_col : str
        Numeric column to plot the KDE for (e.g. 'log_Price').
    group_col : str
        Column to group by (e.g. 'Year' or 'Year_binary').
    xlabel : str, optional
        Label for the x-axis. Defaults to value_col.
    ylabel : str, optional
        Label for the y-axis. Defaults to 'Density'.
    title : str, optional
        Plot title. Defaults to 'Kernel Density Estimate of <value_col> by <group_col>'.
    """
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=df,
        x=value_col,
        hue=group_col,
        fill=True,
        common_norm=False,
        alpha=0.3,
    )

    plt.xlabel(xlabel if xlabel is not None else value_col)
    plt.ylabel(ylabel)
    plt.title(title if title is not None else f"Kernel Density Estimate of {value_col} by {group_col}")
    plt.tight_layout()
    plt.show()



def plot_barchart(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "Count",
) -> None:
    """
    Plot a bar chart of value counts for a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to plot.
    title : str, optional
        Plot title. Defaults to 'Count of <column>'.
    xlabel : str, optional
        x-axis label. Defaults to column name.
    ylabel : str, optional
        y-axis label. Defaults to 'Count'.
    """
    counts = df[column].value_counts().sort_index()

    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")

    plt.ylim(bottom=0)  # force y-axis to start at 0

    plt.title(title if title is not None else f"Count of {column}")
    plt.xlabel(xlabel if xlabel is not None else column)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_kde(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "Density",
) -> None:
    """
    Plot a kernel density estimate (KDE) for a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Numeric column to plot.
    title : str, optional
        Plot title. Defaults to 'KDE of <column>'.
    xlabel : str, optional
        x-axis label. Defaults to column name.
    ylabel : str, optional
        y-axis label. Defaults to 'Density'.
    """
    title = title or f"KDE of {column}"
    xlabel = xlabel or column

    plt.figure(figsize=(8, 4))
    sns.kdeplot(df[column].dropna(), fill=True)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()


def plot_faceted_kde(
    df: pd.DataFrame,
    numeric_col: str,
    facet_col: str,
    col_wrap: int = 4,
    height: float = 2.5,
    aspect: float = 1.2,
) -> None:
    """
    Plot faceted KDE distributions of a numeric variable, split by a categorical variable.

    Args:
        df (pd.DataFrame): Input dataframe.
        numeric_col (str): Numeric column to plot (e.g. YearBuilt).
        facet_col (str): Categorical column to facet by (e.g. CouncilArea).
        col_wrap (int, optional): Number of facets per row. Default is 4.
        height (float, optional): Height of each facet. Default is 2.5.
        aspect (float, optional): Aspect ratio of each facet. Default is 1.2.
    """
    sns.displot(
        data=df.dropna(subset=[numeric_col]),
        x=numeric_col,
        col=facet_col,
        col_wrap=col_wrap,
        kind="kde",
        fill=True,
        height=height,
        aspect=aspect,

    )

    plt.tight_layout()
    plt.show()


def plot_missing_proportion(
    df: pd.DataFrame,
    group_col: str,
    missing_col: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot the proportion of missing values in a column, grouped by another column.

    Args:
        df (pd.DataFrame): Input dataframe.
        group_col (str): Column to group by (e.g. CouncilArea).
        missing_col (str): Binary missingness indicator (e.g. Missing_YearBuilt).
        title (str, optional): Plot title.
    """
    prop_missing = (
        df.groupby(group_col)[missing_col]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 5))
    prop_missing.plot(kind="bar")

    plt.ylim(0, 1)
    plt.ylabel("Proportion Missing")
    plt.xlabel(group_col)
    plt.title(title or f"Proportion of Missing {missing_col.replace('Missing_', '')} by {group_col}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_size_pairplot(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Plot pairwise scatter plots (and histograms on the diagonal)
    for selected size-related features.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list[str]): Columns to include in the pairplot.
    """
    sns.pairplot(
        df[columns].dropna(),
        diag_kind="hist"
    )

    plt.show()