"""Class for selecting most relevant features using the mrmr algorithm."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..featuristic_lib import runMRMRArray
from .utils import as_fortran_xy

# set the floor value for the correlation matrix
FLOOR: float = 0.00001


class MaxRelevanceMinRedundancy:
    """
    Class for selecting most relevant features using the mrmr algorithm.
    """

    def __init__(self, k: int = 6, problem_type: str = "regression"):
        """
        Initialize the MaxRelevanceMinRedundancy class.

        Parameters
        ----------
        K : int (default=6)
            The number of features to select.

        problem_type : str (default='regression')
            The type of problem. Either 'regression' or 'classification'.

        """
        self.k = k
        self.problem_type = problem_type
        self.selected_features = None

        if problem_type not in ["regression", "classification"]:
            raise ValueError(
                "Invalid type. Must be either 'regression' or 'classification'."
            )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the mRMR algorithm to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        Returns
        -------
        None
        """
        self.selected_features = self._mrmr(X, y)

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Transform the data using the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable. Not used in this function.

        Returns
        -------
        pd.DataFrame
            The dataframe with the selected features.
        """
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the mRMR algorithm to the data and transform the data using the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        Returns
        -------
        pd.DataFrame
            The dataframe with the selected features.
        """
        self.fit(X, y)
        return self.transform(X)

    def _mrmr(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        """
        Select the top n_features features using the mRMR algorithm.

        Uses Nim implementation (Fortran-contiguous float64 arrays).

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        Returns
        -------
        list
            The list of selected features.
        """
        target_values = np.asarray(y)
        if target_values.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if len(target_values) != len(X):
            raise ValueError("X and y must have the same number of rows")
        if pd.isna(target_values).any():
            raise ValueError("y must not contain missing values")
        if (
            np.issubdtype(target_values.dtype, np.number)
            and not np.isfinite(target_values).all()
        ):
            raise ValueError("y must contain only finite values")

        # Filter out constant features and features with NaN
        X = X.loc[:, X.nunique() > 1].dropna(axis=1)

        # Set the maximum only after filtering; a candidate pool can become
        # empty when every generated feature is constant or invalid.
        k: int = min(self.k, X.shape[1])
        if k == 0:
            return []

        metric_type = 0
        num_classes = 0
        target = y
        if self.problem_type == "classification":
            encoder = LabelEncoder()
            target = encoder.fit_transform(y)
            num_classes = len(encoder.classes_)
            if num_classes < 2:
                raise ValueError(
                    "Classification target must contain at least two classes"
                )
            metric_type = 1

        X_f, y_c = as_fortran_xy(X, target)
        selected_indices = runMRMRArray(X_f, y_c, k, FLOOR, metric_type, num_classes)
        return [X.columns[i] for i in selected_indices]
