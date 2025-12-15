from unittest.mock import Base
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


"""
Creating a transformer which is a simplified re-implementaion of existing scikit-learn transformer
The Binarizer is a transformer where for threshold k, 
    if values > k, value of feature maps to 1
    if values <=k, value of feature maps to 0 

"""
    
class Binarizer