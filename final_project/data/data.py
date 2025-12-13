from pathlib import Path
from typing import Iterable

import pandas as pd

raw_data = Path(__file__).resolve().parents[2] / "data" / "raw_data.csv"

def load_data() -> pd.DataFrame:
    """Load raw data from raw_data.csv.

    Returns:
        pd.DataFrame: Raw data as a pandas DataFrame

    """
    df = pd.read_csv(raw_data)

    return df


def summarise_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table describing each column of the DataFrame."""

    summary = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "n_missing": df.isna().sum(),
            "pct_missing": df.isna().mean().round(3),
            "n_unique": df.nunique(),
            "max": df.max(numeric_only=True),
            "min": df.min(numeric_only=True),
            "mean": df.mean(numeric_only=True),
            "median": df.median(numeric_only=True),
            "std": df.std(numeric_only=True),
        }
    )

    return summary


def date_cleaning(df: pd.DataFrame, date_col="Date") -> pd.DataFrame:
    """
    Convert a string-based date column into a datetime column and extract
    the Year, Month, and Day components into new columns.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str, optional
        The name of the column containing date strings in the format '%d/%m/%Y'.
        Default is "Date".

    Returns
    -------
    pd.DataFrame
        The dataframe with an updated datetime column and newly added
        'Year', 'Month', and 'Day' integer columns.

    Notes
    -----
    - This function mutates the input dataframe.
    - Assumes date format is day/month/year (e.g., '3/12/2016').
    - Will raise a ValueError if the date format is incorrect.
    """
    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="raise")

    # Extract components
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day

    return df


def float_to_integer(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert a float column to an integer column."""
    df[column] = df[column].astype("Int64")
    return df

import pandas as pd



def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every column in the dataframe that contains missing values,
    create a new indicator feature called 'Missing_<column>'.

    The indicator is:
        1 if the value is missing
        0 otherwise

    The function returns a new dataframe with added columns.
    """

    df = df.copy()  # do not overwrite original unless intended

    for col in df.columns:
        if df[col].isna().any():   # only for columns with missing values
            df[f"Missing_{col}"] = df[col].isna().astype(int)

    return df

import pandas as pd




def impute_council_from_suburb(
    df: pd.DataFrame,
    suburb_col: str = "Suburb",
    council_col: str = "CouncilArea",
    unavailable_label: str = "Unavailable",
) -> pd.DataFrame:
    """
    Impute missing council values using suburb-to-council relationships
    learned from existing data. Remaining missing councils are set to 'Unavailable'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    suburb_col : str, optional
        Name of the suburb column. Default is 'Suburb'.
    council_col : str, optional
        Name of the council column. Default is 'CouncilArea'.
    unknown_label : str, optional
        Label used when council cannot be inferred. Default is 'Unavailable'.

    Returns
    -------
    pd.DataFrame
        DataFrame with CouncilArea imputed.
    """
    df_imputed = df.copy()

    # Build Suburb -> Council mapping from non-missing values
    suburb_to_council = (
        df_imputed.loc[df_imputed[council_col].notna(), [suburb_col, council_col]]
        .groupby(suburb_col)[council_col]
        .agg(lambda x: x.mode().iloc[0])
    )

    # Impute missing councils using suburb mapping
    missing_mask = df_imputed[council_col].isna()
    df_imputed.loc[missing_mask, council_col] = (
        df_imputed.loc[missing_mask, suburb_col].map(suburb_to_council)
    )

    # Any councils still missing -> Unavailable
    df_imputed[council_col] = df_imputed[council_col].fillna(unavailable_label)

    return df_imputed



def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Drop specified columns from a DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (Iterable[str]): Columns to drop.

    Returns:
        pd.DataFrame: DataFrame with columns removed.
    """
    return df.drop(columns=list(columns))