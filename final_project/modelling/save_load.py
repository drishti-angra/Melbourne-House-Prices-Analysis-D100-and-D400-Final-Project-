from pathlib import Path

import joblib
import pandas as pd


def save_pipeline(pipeline, name: str) -> None:
    """
    Save pipeline to the project directory.


    Parameters
    ----------
    pipeline
        pipline to save
    name : str
        Filename

    Returns
    -------
    None
    """
    project_root = Path(__file__).resolve().parents[2]
    pipelines_dir = project_root / "pipelines"

    pipelines_dir.mkdir(exist_ok=True)

    file_path = pipelines_dir / f"{name}.joblib"
    joblib.dump(pipeline, file_path)


def load_pipeline(name: str):
    """
    Load a pipeline from directory.
    """
    project_root = Path(__file__).resolve().parents[2]
    pipelines_dir = project_root / "pipelines"

    file_path = pipelines_dir / f"{name}.joblib"


    return joblib.load(file_path)



def save_X_y(X: pd.DataFrame, y: pd.Series, split_name: str) -> None:
    """
    Save X and y to director.


    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    split_name : str
        Name of the split (e.g. "train", "test")
    """
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    data_dir.mkdir(exist_ok=True)

    X.to_parquet(data_dir / f"X_{split_name}.parquet", index=False)


    y.to_frame(name=y.name).to_parquet(
        data_dir / f"y_{split_name}.parquet",
        index=False,
    )


def load_X_y(split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load X and y from directory.
    """
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    X = pd.read_parquet(data_dir / f"X_{split_name}.parquet")
    y_df = pd.read_parquet(data_dir / f"y_{split_name}.parquet")

    y = y_df.iloc[:, 0]
    y.name = y_df.columns[0]

    return X, y