from preprocessing.categorical import OneHotEncoding
from preprocessing.numerical import Scaler
from preprocessing.selector import Selector

from sklearn.pipeline import FeatureUnion, Pipeline


class Preprocessor:
    def preprocess(self):

        # TODO: split num_pipeline, cat_pipeline
        # into separate functions

        # Set numerical pipeline

        num_pipeline = Pipeline(
            steps=[
                ("selector", Selector(["num1", "num2"])),
                ("scaler", Scaler(with_mean=True, with_std=True)),
            ]
        )

        # Set categorical pipeline

        cat_pipeline = Pipeline(
            steps=[("selector", Selector(["cat1", "cat2"])), ("ohe", OneHotEncoding())]
        )

        feature_union = FeatureUnion(
            transformer_list=[
                ("numerical_pipeline", num_pipeline),
                ("categorical_pipeline", cat_pipeline),
            ]
        )

        return feature_union
