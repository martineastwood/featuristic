"""Class for selecting most relevant features using the mrmr algorithm."""

from typing import Iterable
import pandas as pd
from sklearn.feature_selection import f_regression, f_classif
from tqdm import tqdm


# set the floor value for the correlation matrix
FLOOR: float = 0.00001


class MaxRelevanceMinRedundancy:
    """
    Class for selecting most relevant features using the mrmr algorithm.
    """

    def __init__(self, k: int = 6, type: str = "regression"):
        """
        Initialize the MaxRelevanceMinRedundancy class.

        Parameters
        ----------
        K : int (default=6)
            The number of features to select.
        """
        self.k = k
        self.selected_features = None

        if type not in ["regression", "classification"]:
            raise ValueError(
                "Invalid type. Must be either 'regression' or 'classification'."
            )
        if type == "regression":
            self.metric = f_regression
        else:
            self.metric = f_classif

    def fit(self, X, y: Iterable):
        """
        Fit the mRMR algorithm to the data.

        Parameters
        ----------
        X : MatrixType
            The dataframe with the features.
        y : Iterable
            The target variable.

        Returns
        -------
        None
        """
        self.selected_features = self._mrmr(X, y)

    def transform(self, X):
        """
        Transform the data using the selected features.

        Parameters
        ----------
        X : MatrixType
            The dataframe with the features.

        Returns
        -------
        MatrixType
            The MatrixType with the selected features.
        """
        return X[self.selected_features]

    def fit_transform(self, X, y: Iterable):
        """
        Fit the mRMR algorithm to the data and transform the data using the selected features.

        Parameters
        ----------
        X : MatrixType
            The MatrixType with the features.
        y : Iterable
            The target variable.

        Returns
        -------
        MatrixType
            The dataframe with the selected features.
        """
        self.fit(X, y)
        return self.transform(X)

    def _mrmr(self, X, y: Iterable):
        """
        Select the top n_features features using the mRMR algorithm.

        Parameters
        ----------
        X : MatrixType
            A pandas dataframe containing the features to prune.
        y : Iterable
            The target variable.

        Returns
        -------
        list
            The list of selected features.
        """
        # set the maximum number of features to select
        k: int = min(self.k, X.shape[1])

        # calculate the f-statistic and the correlation matrix
        f_stat = pd.Series(self.metric(X, y)[0], index=X.columns)
        corr = pd.DataFrame(FLOOR, index=X.columns, columns=X.columns)

        # initialize list of selected features and list of excluded features
        selected = []
        not_selected = X.columns.to_list()

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
            pbar.update(1)

        return selected
