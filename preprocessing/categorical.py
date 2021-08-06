from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder


class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ohe = OneHotEncoder()

    def fit(self, X, y=None):
        self.ohe.fit(X)
        return self

    def transform(self, X, y=None):
        X_transformed = self.ohe.transform(X)
        return X_transformed
