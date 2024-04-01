import featuristic as ft
import pytest
import pandas as pd


def test_gfs():
    num_features = 5
    gfs = ft.GeneticFeatureSynthesis(num_features=num_features, pbar=False)

    with pytest.raises(Exception):
        gfs.plot_history()

    assert gfs.len_hall_of_fame == num_features * 5
    assert gfs.fit_called == False

    with pytest.raises(Exception):
        gfs.fit(X=None, y=None)

    X = pd.DataFrame({"a": [1, 2, 4], "b": [4, 5, 6]})
    y = pd.Series([1, 2, 3])

    gfs.fit(X, y)
    new_X = gfs.transform(X)
    new_cols = [x for x in new_X.columns if x.startswith("feature_")]
    assert len(new_cols) == num_features
    assert gfs.fit_called == True

    gfs = ft.GeneticFeatureSynthesis(num_features=num_features, pbar=False)
    new_X = gfs.fit_transform(X, y)
    new_cols = [x for x in new_X.columns if x.startswith("feature_")]
    assert len(new_cols) == num_features
