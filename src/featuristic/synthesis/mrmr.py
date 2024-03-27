"""Class for selecting most relevant features using the mrmr algorithm."""

from typing import List

import pandas as pd
from sklearn.feature_selection import f_classif, f_regression
from tqdm import tqdm

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

        X = X.loc[:, X.nunique() > 1]

        # calculate the f-statistic and the correlation matrix
        f_stat = pd.Series(self.metric(X, y)[0], index=X.columns)
        corr = pd.DataFrame(FLOOR, index=X.columns, columns=X.columns)

        # initialize list of selected features and list of excluded features
        selected = []
        not_selected = X.columns.to_list()

        if self.pbar:
            pbar = tqdm(total=k, desc="Pruning feature space...")

        # select the top K features
        for i in range(k):
            if i > 0:
                last_selected = selected[-1]
                corr.loc[not_selected, last_selected] = (
                    X[not_selected].corrwith(X[last_selected]).abs().clip(FLOOR)
                )

            score = f_stat.loc[not_selected] / corr.loc[not_selected, selected].mean(
                axis=1
            ).fillna(FLOOR)

            best = score.index[score.argmax()]
            selected.append(best)
            not_selected.remove(best)

            if self.pbar:
                pbar.update(1)

        return selected
