from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


import pandas as pd
import hashlib
import numpy as np 




cleaned_data = Path(__file__).resolve().parents[2] / "data" / "cleaned_data.parquet"

def load_parquet_data() -> pd.DataFrame:
    """
    Load cleaned data from a parquet file.
    
    Returns:
    pd.DataFrame: Cleaned data as a pandas DataFrame
    """

    df = pd.read_parquet(cleaned_data)
    return df



def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    if df[id_column].dtype == np.int64:
        modulo = df[id_column] % 100
    else:
        modulo = df[id_column].apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 100
        )

    df["sample"] = np.where(modulo < training_frac * 100, "train", "test")

    return df


def save_train_test_parquet(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and test sets using the 'sample' column
    and save them to the project's data directory as parquet files.

    The files are saved to:
    house_prices_final_project/data/train.parquet
    house_prices_final_project/data/test.parquet

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    data_dir.mkdir(exist_ok=True)

    train_df = df[df["sample"] == "train"].drop(columns="sample")
    test_df = df[df["sample"] == "test"].drop(columns="sample")

    train_df.to_parquet(data_dir / "train.parquet", index=False)
    test_df.to_parquet(data_dir / "test.parquet", index=False)

    return train_df, test_df

class ZeroMedian_Imputer(BaseEstimator, TransformerMixin):
    """
    Imputes median for values which are zero
    """

    def fit(self, X, y=None):

        X = np.asarray(X, dtype=float)
        X_new = np.where(X ==0, np.nan, X)
        self.medians_ = np.nanmedian(X_new, axis=0)
        return self
    
    def transform(self, X):

        X = np.asarray(X, dtype=float).copy()

        for i in range(X.shape[1]):
            X[X[:, i] == 0, i] = self.medians_[i]

        return X
    
