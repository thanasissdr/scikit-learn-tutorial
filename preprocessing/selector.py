from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]
