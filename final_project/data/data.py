from pathlib import Path
from typing import Iterable

import pandas as pd

raw_data = Path(__file__).resolve().parents[2] / "data" / "raw_data.csv"


def load_data() -> pd.DataFrame:
    """Load raw data.

    Returns:
        pd.DataFrame: Raw data as pandas DataFrame

    """
    df = pd.read_csv(raw_data)

    return df


def summarise_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table describing columns to see dtype, missing data and summary statistics."""

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
    Convert a date column into a datetime column and extract the Year, Month, and Day components.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str, optional

    Returns
    -------
    pd.DataFrame
        new dataframe with newly added
        'Year', 'Month', and 'Day' integer columns.

    """

    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="raise")


    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day

    return df


def float_to_integer(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert a float column to integer column."""
    df[column] = df[column].astype("Int64")
    return df




def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every column in the dataframe that contains missing values,
    create a indicator feature: 
        1 if missing
        0 otherwise

    Returns
    -----
    dataframe with the binary indicators.
    """

    df = df.copy() 

    for col in df.columns:
        if df[col].isna().any():  
            df[f"Missing_{col}"] = df[col].isna().astype(int)

    return df





def impute_council_from_suburb(
    df: pd.DataFrame,
    suburb_col: str = "Suburb",
    council_col: str = "CouncilArea",
    unavailable_label: str = "Unavailable",
) -> pd.DataFrame:
    """
    Impute missing council values using suburb-to-council relationships
    learned from existing data. Remaining missing councils are 'Unavailable'.

    Parameters
    ----------
    df : pd.DataFrame
    suburb_col : str, optional
        Name of suburb column
    council_col : str, optional
        Name of council colum
    unknown_label : str, optional
        Label used when council not possible to be imputed. Default is 'Unavailable'.

    Returns
    -------
    pd.DataFrame
        DataFrame with CouncilArea imputed.
    """
    df_imputed = df.copy()

    suburb_to_council = (
        df_imputed.loc[df_imputed[council_col].notna(), [suburb_col, council_col]]
        .groupby(suburb_col)[council_col]
        .agg(lambda x: x.mode().iloc[0])
    )


    missing_mask = df_imputed[council_col].isna()
    df_imputed.loc[missing_mask, council_col] = (
        df_imputed.loc[missing_mask, suburb_col].map(suburb_to_council)
    )


    df_imputed[council_col] = df_imputed[council_col].fillna(unavailable_label)

    return df_imputed



def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Drop specified columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (Iterable[str]): Columns to drop.

    Returns:
        pd.DataFrame: DataFrame with columns dropped.
    """
    return df.drop(columns=list(columns))


cleaned_data = Path(__file__).resolve().parents[2] / "data" / "cleaned_data.parquet"

def save_cleaned_data(df: pd.DataFrame) -> None:
    """Saving cleaned data to cleaned_data.parquet.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """
    cleaned_data.parent.mkdir(exist_ok=True)
    df.to_parquet(cleaned_data, index=False)



def add_zero_dummy(df, column):
    """
    Creates a dummy column where:
    - 1 if df[column] = 0
    - 0 otherwise

    new column named column + '_zero'
    """
    dummy_name = column + "_zero"
    df[dummy_name] = (df[column] == 0).astype(int)
    return df
