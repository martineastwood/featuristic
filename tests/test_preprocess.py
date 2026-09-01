import pandas as pd

import featuristic as ft
from featuristic.synthesis.mrmr import MaxRelevanceMinRedundancy


def test_preprocess_data():
    X = pd.DataFrame({"a": [1, 1, 1], "b": [4, 5, 6]})
    y = pd.Series([1, 2, 3])
    X_new, y_new = ft.synthesis.preprocess.preprocess_data(X, y)
    assert X_new.columns.tolist() == ["b"]
    assert y_new.tolist() == [1, 2, 3]


def test_mrmr_returns_empty_frame_when_all_features_are_constant():
    X = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    result = MaxRelevanceMinRedundancy(k=1, pbar=False).fit_transform(X, y)

    assert result.empty
    assert result.index.equals(X.index)
