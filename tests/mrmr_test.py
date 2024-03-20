import numpy as np
import pandas as pd
import pytest
from sklearn import datasets as sklearn_datasets

import featuristic


@pytest.fixture
def data_pandas():
    data = sklearn_datasets.load_wine(as_frame=True)
    return data["data"], data["target"]


@pytest.fixture
def data_numpy():
    data = sklearn_datasets.load_wine(as_frame=False)
    return data["data"], data["target"]


def test_with_pandas_df(data_pandas):
    df, target = data_pandas
    assert df.shape[0] == target.shape[0]

    mrmr = featuristic.selection.MaxRelevanceMinRedundancy(k=3)
    mrmr.fit(df, target)
    feats = mrmr.transform(df)

    assert feats.shape[0] == target.shape[0]
    assert feats.shape[1] == 3
    assert all([col in df.columns for col in feats.columns])
    assert isinstance(feats, pd.DataFrame)


def test_with_numpy(data_numpy):
    df, target = data_numpy
    assert df.shape[0] == target.shape[0]

    mrmr = featuristic.selection.MaxRelevanceMinRedundancy(k=3)
    mrmr.fit(df, target)
    feats = mrmr.transform(df)

    assert feats.shape[0] == target.shape[0]
    assert feats.shape[1] == 3
    assert isinstance(feats, np.ndarray)
