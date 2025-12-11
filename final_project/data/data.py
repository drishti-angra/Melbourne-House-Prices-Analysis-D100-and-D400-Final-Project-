from pathlib import Path
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
    
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": df.isna().mean().round(3),
        "n_unique": df.nunique(),
        "max": df.max(numeric_only=True),
        "min": df.min(numeric_only=True),
        "mean": df.mean(numeric_only=True),
        "median": df.median(numeric_only=True),
        "std": df.std(numeric_only=True)
        
    })
    
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