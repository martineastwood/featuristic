import featuristic as ft
import pandas as pd


def test_fetch_cars():
    X, y = ft.fetch_cars_dataset()

    assert X.shape == (392, 7)
    assert y.shape == (392,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_wine_cars():
    X, y = ft.fetch_wine_dataset()

    assert X.shape == (178, 13)
    assert y.shape == (178,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
