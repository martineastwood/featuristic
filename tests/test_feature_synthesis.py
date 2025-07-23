import pandas as pd
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from featuristic import FeatureSynthesis


@pytest.fixture
def regression_data():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] * 2 + X["x1"] - X["x2"]
    return X, y


def test_feature_synthesis_fit_transform(regression_data):
    X, y = regression_data
    fs = FeatureSynthesis(
        num_features=3, max_generations=5, population_size=20, n_jobs=1, pbar=False
    )
    Xt = fs.fit_transform(X, y)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == X.shape[0]
    assert fs.fit_called
    assert fs.get_feature_info().shape[0] == 3


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_history_runs():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] + X["x1"] * 2 - X["x2"]

    fs = FeatureSynthesis(
        num_features=2, max_generations=3, population_size=10, n_jobs=1, pbar=False
    )
    fs.fit(X, y)

    try:
        fs.plot_history()
    except Exception as e:
        assert False, f"plot_history raised an exception: {e}"
