import featuristic as ft
import pandas as pd


def test_preprocess():
    X = pd.DataFrame({"a": [1, 1, 1], "b": [4, 5, 6]})
    X_new = ft.synthesis.preprocess.drop_low_variance_features(X, threshold=0.1)
    assert X_new.columns.tolist() == ["b"]


def test_preprocess_data():
    X = pd.DataFrame({"a": [1, 1, 1], "b": [4, 5, 6]})
    y = pd.Series([1, 2, 3])
    X_new, y_new = ft.synthesis.preprocess.preprocess_data(X, y)
    assert X_new.columns.tolist() == ["b"]
    assert y_new.tolist() == [1, 2, 3]
