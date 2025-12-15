from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder 
from typing import Tuple



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
  



def split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into X_train, y_train, X_test, y_test.
    Dropping Address as it is a string.

    Parameters
    ----------
    df : pd.DataFrame (Input data)
    target_col : str (Name of target column)

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """

    X = df.drop(columns=[target_col, "Address"])
    y = df[target_col]

    return X, y


class ZeroMedianImputer(BaseEstimator, TransformerMixin):
    """Impute median for values equal to 0."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        X_no_zero = np.where(X == 0, np.nan, X)
        self.medians_ = np.nanmedian(X_no_zero, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "medians_")
        X = np.asarray(X, dtype=float)
        return np.where(X == 0, self.medians_, X)



class UpperWinsorizer(BaseEstimator, TransformerMixin):
    """
    Upper tail winsorization transformer.
    """
    def __init__(self, upper_quantile=0.95):
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.upper_bound_ = np.percentile(X, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "upper_bound_")
        return np.minimum(X, self.upper_bound_)



class ConditionalMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputes median values conditional on a feature.
    """

    def __init__(self, target_col: str, condition_col: str):
        self.target_col = target_col
        self.condition_col = condition_col

    def fit(self, X, y=None):
        self.condition_medians_ = (
            X.groupby(self.condition_col)[self.target_col].median()
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "condition_medians_")

        X = X.copy()

        X[self.target_col] = X[self.target_col].astype(float)

        missing = X[self.target_col].isna()

        if missing.any():
            X.loc[missing, self.target_col] = (
                X.loc[missing, self.condition_col].map(self.condition_medians_)
            )

        return X


def preprocessor() -> ColumnTransformer:

    #pipleline for Bathroom, Bedroom2
    rooms_pipeline = Pipeline(
        steps = [
            ("zero_median_imputer", ZeroMedianImputer()),
        ]
    )
    #pipeline for Landsize
    landsize_pipeline = Pipeline(
        steps = [
            ("upper_tail_winsorizer", UpperWinsorizer(upper_quantile=0.95)),
            ("standard_scaler", StandardScaler()),
        ]
    )

    #pipeline for Car
    car_pipeline = Pipeline(
        steps = [
            ("median_imputer", SimpleImputer(strategy="median")),
        ]
    )

    #pipeline for YearBuilt 
    yearbuilt_pipeline = Pipeline(
        steps = [
            ("median_imputer", SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler()),
        ]
    )

    #pipeline for Method, Regionname, Type
    categorical_pipeline = Pipeline(
        steps = [
            ("onehot_encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )


    #pipeline for Suburb
    suburb_pipeline = Pipeline(
        steps = [
            ("target_encoder", TargetEncoder(smoothing=10)),
        ]
    )


    #pipeline for Longtitude, Lattitude
    location_pipeline = Pipeline(
        steps = [
            ("standard_scaler", StandardScaler()),
            ("spline_transformer", SplineTransformer(knots="quantile", include_bias=False, n_knots=5)),
            
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("rooms", rooms_pipeline, ["Bathroom", "Bedroom2"]),
            ("landsize", landsize_pipeline, ["Landsize"]),
            ("car", car_pipeline, ["Car"]),
            ("yearbuilt", yearbuilt_pipeline, ["YearBuilt"]),
            ("categorical", categorical_pipeline, ["Method", "Regionname", "Type"]),
            ("suburb", suburb_pipeline, ["Suburb"]),
            ("location", location_pipeline, ["Longtitude", "Lattitude"]),
        ],
        remainder="passthrough"
    )

    return preprocessor

