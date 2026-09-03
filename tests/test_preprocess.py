import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import f_classif, f_regression

import featuristic as ft
from featuristic.synthesis.mrmr import FLOOR, MaxRelevanceMinRedundancy


def test_preprocess_data():
    X = pd.DataFrame({"a": [1, 1, 1], "b": [4, 5, 6]})
    y = pd.Series([1, 2, 3])
    X_new, y_new = ft.synthesis.preprocess.preprocess_data(X, y)
    assert X_new.columns.tolist() == ["b"]
    assert y_new.tolist() == [1, 2, 3]


def test_mrmr_returns_empty_frame_when_all_features_are_constant():
    X = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    result = MaxRelevanceMinRedundancy(k=1).fit_transform(X, y)

    assert result.empty
    assert result.index.equals(X.index)


def test_mrmr_classification_uses_anova_relevance():
    y = pd.Series([0] * 4 + [1] * 4 + [2] * 4)
    X = pd.DataFrame(
        {
            # Strong class separation but no linear relationship with class label.
            "class_signal": [
                10.0,
                10.1,
                9.9,
                10.0,
                0.0,
                0.1,
                -0.1,
                0.0,
                10.0,
                10.1,
                9.9,
                10.0,
            ],
            "linear_signal": [
                0.0,
                0.2,
                -0.1,
                0.1,
                1.0,
                1.2,
                0.9,
                1.1,
                2.0,
                2.2,
                1.9,
                2.1,
            ],
        }
    )

    result = MaxRelevanceMinRedundancy(
        k=1, problem_type="classification"
    ).fit_transform(X, y)

    assert result.columns.tolist() == ["class_signal"]


@pytest.mark.parametrize(
    "labels",
    [
        ["negative"] * 4 + ["positive"] * 4,
        ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
    ],
)
def test_native_classification_mrmr_matches_sklearn_top_feature(labels):
    rows = len(labels)
    class_codes = pd.factorize(pd.Series(labels))[0]
    X = pd.DataFrame(
        {
            "class_signal": class_codes + [0.0, 0.1, -0.1, 0.05] * (rows // 4),
            "noise": [0.2, -0.3, 0.1, 0.4] * (rows // 4),
        }
    )
    expected = X.columns[f_classif(X, labels)[0].argmax()]

    result = MaxRelevanceMinRedundancy(
        k=1, problem_type="classification"
    ).fit_transform(X, pd.Series(labels))

    assert result.columns.tolist() == [expected]


def test_native_regression_mrmr_matches_sklearn_selection():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(60, 8)), columns=list("abcdefgh"))
    y = pd.Series(rng.normal(size=60))
    relevance = pd.Series(f_regression(X, y)[0], index=X.columns)
    correlations = pd.DataFrame(FLOOR, index=X.columns, columns=X.columns)
    expected = []
    not_selected = X.columns.to_list()
    for _ in range(5):
        if expected:
            correlations.loc[not_selected, expected[-1]] = (
                X[not_selected].corrwith(X[expected[-1]]).abs().clip(lower=FLOOR)
            )
        redundancy = correlations.loc[not_selected, expected].mean(axis=1).fillna(FLOOR)
        best = (relevance.loc[not_selected] / redundancy).idxmax()
        expected.append(best)
        not_selected.remove(best)

    result = MaxRelevanceMinRedundancy(k=5).fit_transform(X, y)

    assert result.columns.tolist() == expected
