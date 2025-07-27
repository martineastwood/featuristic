from typing import List, Callable, Tuple, Optional

import pandas as pd
from sklearn.feature_selection import f_classif, f_regression
from sklearn.utils.multiclass import type_of_target
from tqdm import tqdm
import numpy as np

FLOOR: float = 1e-5  # Prevent division by zero and log(0)


class MaxRelevanceMinRedundancy:
    """
    Selects k features that are maximally relevant to the target and minimally redundant.
    """

    def __init__(self, k: int = 6, show_progress_bar: bool = True):
        """
        Initialize the MaxRelevanceMinRedundancy class.

        Parameters
        ----------
        k : int (default=6)
            The number of features to select.

        show_progress_bar : bool (default=True)
            Whether to display a progress bar using tqdm.
        """
        self.k = k
        self.show_progress_bar = show_progress_bar
        self.selected_features_: List[str] = None
        self.metric: Optional[
            Callable[[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray]]
        ] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the mRMR selector to the data.

        This method identifies the optimal set of features based on the mRMR criteria.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe with the features.
        y : pd.Series
            The target variable.
        """
        if self.metric is None:
            target_type = type_of_target(y)
            if target_type in ("binary", "multiclass", "multiclass-multioutput"):
                self.metric = f_classif
            elif target_type in ("continuous", "continuous-multioutput"):
                self.metric = f_regression
            else:
                raise ValueError(f"Unsupported target type: {target_type}")

        self.selected_features_ = self._mrmr(X, y)

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transform the data using the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series, optional
            The target variable. Not used in this function, kept for API consistency.

        Returns
        -------
        pd.DataFrame
            The dataframe with the selected features.
        """
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the mRMR algorithm and transform the data.

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

    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the data by removing unsuitable columns."""
        # Drop duplicate columns
        X = X.loc[:, ~X.T.duplicated()]
        # Drop columns with only one unique value and rows with NaN
        X = X.loc[:, X.nunique() > 1].dropna(axis=1)
        return X

    def _calculate_relevance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculates the relevance of each feature to the target."""
        f_stat, _ = self.metric(X, y)
        relevance = pd.Series(f_stat, index=X.columns)
        relevance = relevance.replace([np.inf, -np.inf], np.finfo(np.float64).max)
        # f_regression can return NaN for perfectly correlated features.
        # We treat this as maximum relevance.
        relevance = relevance.fillna(np.finfo(np.float64).max)
        return relevance

    def _calculate_redundancy_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculates the redundancy matrix between features."""
        corr = X.corr().abs().clip(lower=FLOOR)
        np.fill_diagonal(corr.values, 0)  # Set self-correlation to 0
        return corr

    def _select_features(
        self, relevance: pd.Series, redundancy_matrix: pd.DataFrame, num_features: int
    ) -> List[str]:
        """Selects the best features based on relevance and redundancy."""
        k = min(self.k, num_features)
        if k == 0:
            return []
        features = relevance.index.to_list()
        selected = []

        if not features:
            return []

        # Select the first feature based on the highest relevance
        first_feature = relevance.idxmax()
        selected.append(first_feature)
        features.remove(first_feature)

        with tqdm(
            total=k, desc="Pruning feature space...", disable=not self.show_progress_bar
        ) as pbar:
            pbar.update(1)
            for _ in range(min(k - 1, len(features))):
                scores = pd.Series(index=features, dtype=float)
                for feature in features:
                    # Calculate the redundancy of the feature with the already selected features
                    redundancy = redundancy_matrix.loc[feature, selected].mean()
                    # The score is the relevance divided by the redundancy
                    scores[feature] = relevance[feature] / (redundancy + FLOOR)

                best = scores.idxmax()
                selected.append(best)
                features.remove(best)
                pbar.update(1)

        return selected

    def _mrmr(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Selects the best features using the mRMR algorithm.

        This method orchestrates the feature selection process by:
        1. Preprocessing the data.
        2. Calculating feature relevance.
        3. Calculating feature redundancy.
        4. Iteratively selecting features that maximize relevance and minimize redundancy.
        """
        X = self._preprocess_data(X)

        if X.empty:
            return []

        relevance = self._calculate_relevance(X, y)
        X = X[relevance.index]

        redundancy_matrix = self._calculate_redundancy_matrix(X)

        selected_features = self._select_features(
            relevance, redundancy_matrix, X.shape[1]
        )

        return selected_features
