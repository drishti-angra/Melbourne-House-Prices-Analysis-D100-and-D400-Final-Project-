from pathlib import Path
import pandas as pd


cleaned_data = Path(__file__).resolve().parents[2] / "data" / "cleaned_data.parquet"

def load_parquet_data() -> pd.DataFrame:
    """
    Load cleaned data from a parquet file.
    
    Returns:
    pd.DataFrame: Cleaned data as a pandas DataFrame
    """

    df = pd.read_parquet(cleaned_data)
    return df



