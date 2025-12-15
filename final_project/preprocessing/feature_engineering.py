import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


"""
Creating a transformer which is a simplified re-implementaion of existing scikit-learn transformer (Binarizer)
The Binarizer is a transformer where for threshold k, 
    if values > k, value of feature maps to 1
    if values <=k, value of feature maps to 0 

"""
    

class CustomBinarizer(BaseEstimator, TransformerMixin):
    """
    Binarizer transformer where for threshold k:
        if values > k, value of feature maps to 1
        if values <=k, value of feature maps to 0 
    """

    def __init__(self, threshold: float, columns: list[str]):
        self.threshold = threshold
        self.columns = columns

    def fit(self, X, y=None):
        self.names_in_ = self.columns
        return self

    def transform(self, X):
        check_is_fitted(self, "names_in_")

        X = X.copy()

        for col in self.columns:
            X[col] = (X[col] > self.threshold).astype(int)

        return X


