import numpy as np
import pandas as pd


from preprocessing.preprocessor import Preprocessor

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from typing import Dict, Union


def create_df(N: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    # Create numerical columns
    num1 = np.random.uniform(size=(N,))
    num2 = np.random.normal(loc=100, scale=10, size=(N,))

    # Create categorical columns
    cat1 = np.random.choice(["m", "f"], p=[0.8, 0.2], size=(N,))
    cat2 = np.random.choice(["a", "b", "c", "d"], size=(N,))

    label = np.random.choice([0, 1], p=[0.7, 0.3], size=(N,))

    # Return dataframe
    df = pd.DataFrame(
        {"num1": num1, "num2": num2, "cat1": cat1, "cat2": cat2, "label": label}
    )
    return df


class Main:
    def run(
        self,
        estimator,
        preprocessing_pipeline,
        train_test_split_kwargs=None,
    ):
        # Get data
        df = create_df(2000)

        X = df.drop("label", axis=1) # feature matrix
        y = df["label"]

        # Split train, test
        X_train, X_test, y_train, y_test = self._split_train_test(
            X, y, train_test_split_kwargs
        )

        # Preprocess X_train, X_test
        X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
        X_test_preprocessed = preprocessing_pipeline.transform(X_test)

        # fit training data to estimator
        fitted_estimator = estimator.fit(X_train_preprocessed, y_train)

        # assess performance of the fitted estimator
        #  on development(dev)/validation set
        y_pred = fitted_estimator.predict(X_test_preprocessed)
        return classification_report(y_true=y_test, y_pred=y_pred)

    def _split_train_test(self, X, y, train_test_split_kwargs=Union[Dict, None]):
        if train_test_split_kwargs is None:
            train_test_split_kwargs = {}

        train_test_split_kwargs["stratify"] = (
            y if train_test_split_kwargs.get("stratify") == True else None
        )
        if train_test_split_kwargs["stratify"] is not None:
            print("Stratification applied\n")
        else:
            print("Stratification NOT applied\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **train_test_split_kwargs
        )

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    PREPROCESSING_PIPELINE = Preprocessor().preprocess()
    TRAIN_TEST_SPLIT_KWARGS = {"stratify": True, "test_size": 0.2, "random_state": 42}
    ESTIMATOR_KWARGS = {"random_state": 42, "penalty": "l2"}
    ESTIMATOR = LogisticRegression(**ESTIMATOR_KWARGS)

    print(Main().run(ESTIMATOR, PREPROCESSING_PIPELINE, TRAIN_TEST_SPLIT_KWARGS))
