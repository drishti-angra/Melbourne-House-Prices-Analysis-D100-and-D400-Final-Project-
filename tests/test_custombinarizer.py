import numpy as np
import pandas as pd
import pytest

from final_project.preprocessing.feature_engineering import CustomBinarizer



"""
To implement the pytest, type pytest tests/test_custombinarizer.py in the terminal. 
Ensure you are in the parent directory and in the d100_housing virtual env.
"""


@pytest.mark.parametrize(
    "threshold", [-2.0, -1.0, 0.0, 1.0, 2.0]
)
def test_custom_binarizer(threshold):

    # Arrange- setting up the mock dataset which is generating 1000 observations from a standard normal distribution
    X = np.random.normal(0, 1, 1000)
    X_df = pd.DataFrame({"A": X})

    # Act- Applying the transformation on the mock dataset 
    Xt = CustomBinarizer(threshold=threshold, columns=["A"]).fit_transform(X_df)

    # Assert- 
        # checking that the maximum and minimum values for the transformed dataset are 1 and 0. 
        # checking that if X > threshold, 1 is the output. Else, 0 is the output
    assert (
        (Xt["A"].max() == 1)
        & (Xt["A"].min() == 0)
        & ((Xt["A"] == (X > threshold).astype(int)).all())
    )
