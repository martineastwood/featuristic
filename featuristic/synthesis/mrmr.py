"""Class for selecting most relevant features using the mrmr algorithm."""

from typing import List

import pandas as pd
from sklearn.feature_selection import f_classif, f_regression

from ..featuristic_lib import runMRMRZerocopy
from ..synthesis.utils import extract_feature_pointers, extract_target_pointer

# set the floor value for the correlation matrix
FLOOR: float = 0.00001


class MaxRelevanceMinRedundancy:
    """
    Class for selecting most relevant features using the mrmr algorithm.
    """

    def __init__(self, k: int = 6, problem_type: str = "regression", pbar=True):
        """
        Initialize the MaxRelevanceMinRedundancy class.

        Parameters
        ----------
        K : int (default=6)
            The number of features to select.

        problem_type : str (default='regression')
            The type of problem. Either 'regression' or 'classification'.

        pbar : bool (default=True)
            Whether to display a progress bar or not.
        """
        self.k = k
        self.selected_features = None
        self.pbar = pbar

        if problem_type not in ["regression", "classification"]:
            raise ValueError(
                "Invalid type. Must be either 'regression' or 'classification'."
            )
        if problem_type == "regression":
            self.metric = f_regression
        else:
            self.metric = f_classif

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

    def _mrmr(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Select the top n_features features using the mRMR algorithm.

        Uses Nim implementation for 38x speedup via zero-copy NumPy access.

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
        # set the maximum number of features to select
        k: int = min(self.k, X.shape[1])

        # Filter out constant features and features with NaN
        X = X.loc[:, X.nunique() > 1].dropna(axis=1)

        # Extract pointers using centralized backend utilities
        feature_ptrs, _ = extract_feature_pointers(X)
        target_ptr, _ = extract_target_pointer(y)

        # Call Nim mRMR implementation (38x faster)
        selected_indices = runMRMRZerocopy(
            featurePtrs=feature_ptrs,
            targetPtr=target_ptr,
            numRows=len(X),
            numFeatures=len(X.columns),
            k=k,
            floor=FLOOR,
        )

        # Convert indices back to feature names
        selected_features = [str(X.columns[i]) for i in selected_indices]

        return selected_features
