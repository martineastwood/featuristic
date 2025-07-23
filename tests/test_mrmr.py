import pandas as pd
import numpy as np
import pytest
from featuristic.core.mrmr import MaxRelevanceMinRedundancy


def test_mrmr_feature_selection():
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
    y = X["f0"] + X["f1"] * 2

    mrmr = MaxRelevanceMinRedundancy(k=5, pbar=False)
    mrmr.fit(X, y)

    assert len(mrmr.selected_features) == 5
    assert all(f in X.columns for f in mrmr.selected_features)
