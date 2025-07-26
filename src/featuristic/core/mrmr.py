from typing import List

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

    def __init__(self, k: int = 6, pbar: bool = True):
        """
        Initialize the MaxRelevanceMinRedundancy class.

        Parameters
        ----------
        k : int (default=6)
            The number of features to select.

        pbar : bool (default=True)
            Whether to display a progress bar or not.
        """
        self.k = k
        self.pbar = pbar
        self.selected_features = None
        self.metric = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.metric is None:
            target_type = type_of_target(y)
            if target_type in ("binary", "multiclass", "multiclass-multioutput"):
                self.metric = f_classif
            elif target_type in ("continuous", "continuous-multioutput"):
                self.metric = f_regression
            else:
                raise ValueError(f"Unsupported target type: {target_type}")

        self.selected_features = self._mrmr(X, y)

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
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

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
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
        Select the best features using the mRMR algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        Returns
        -------
        List[str]
            The list of selected features.
        """
        # Drop duplicate columns to avoid duplicate feature selection errors
        X = X.loc[:, X.nunique() > 1].dropna(axis=1)
        X = X.loc[:, ~X.T.duplicated()]

        k = min(self.k, X.shape[1])
        if k == 0:
            return []

        # 1. Compute relevance scores
        f_stat = pd.Series(self.metric(X, y)[0], index=X.columns)
        # Remove features with NaN f_stat
        f_stat = f_stat.dropna()
        X = X[f_stat.index]

        # 2. Precompute absolute correlation matrix (redundancy)
        corr = X.corr().abs().clip(lower=FLOOR).astype(np.float64)
        for col in corr.columns:
            corr.loc[col, col] = FLOOR  # Avoid self-correlation = 1

        selected = []
        not_selected = X.columns.to_list()

        if self.pbar:
            pbar = tqdm(total=k, desc="Pruning feature space...")

        for _ in range(k):
            if not selected:
                # First round: no redundancy penalty
                score = f_stat.loc[not_selected]
            else:
                # Compute score = relevance / average redundancy
                redundancy = corr.loc[not_selected, selected].mean(axis=1).fillna(FLOOR)
                score = f_stat.loc[not_selected] / redundancy

            score = score[
                score.index.isin(not_selected)
            ]  # filter to valid candidates only
            if score.empty:
                break
            best = score.idxmax()
            selected.append(best)
            if best in not_selected:
                not_selected.remove(best)

            if self.pbar:
                pbar.update(1)

        if self.pbar:
            if not isinstance(self.pbar, bool):
                self.pbar.close()

        return selected
