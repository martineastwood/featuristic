import pandas as pd
from sklearn.feature_selection import f_regression
from tqdm import tqdm
from featurize.logging import logger


class MaxRelevanceMinRedundancy:
    def __init__(self, K=6):
        """
        Initialize the MaxRelevanceMinRedundancy class.

        Parameters
        ----------
        K : int (default=6)
            The number of features to select.
        """
        logger.info("Initializing MaxRelevanceMinRedundancy class")
        self.K = K

    def fit(self, X, y):
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
        logger.info("Fitting mrmr algorithm to the data")
        self.selected_features = self._mrmr(X, y, self.K)

    def transform(self, X):
        """
        Transform the data using the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe with the features.

        Returns
        -------
        pd.DataFrame
            The dataframe with the selected features.
        """
        logger.info("Transforming the data using the selected features")
        return X[self.selected_features]

    def fit_transform(self, X, y):
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
        logger.info("Fitting and transforming the data using the selected features")
        self.fit(X, y)
        return self.transform(X)

    @staticmethod
    def _mrmr(X, y, K=6):
        """
        Select the top n_features features using the mRMR algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The dataframe with the features.
        y : pd.Series
            The target variable.
        K : int (default=6)
            The number of features to select.

        Returns
        -------
        list
            The list of selected features.
        """
        # set the floor value for the correlation matrix
        FLOOR = 0.00001

        # set the maximum number of features to select
        K = min(K, X.shape[1])
        logger.info(f"Setting mrmr k to {K}")

        # calculate the f-statistic and the correlation matrix
        f_stat = pd.Series(f_regression(X, y)[0], index=X.columns)
        corr = pd.DataFrame(FLOOR, index=X.columns, columns=X.columns)

        # initialize list of selected features and list of excluded features
        selected = []
        not_selected = X.columns.to_list()

        pbar = tqdm(total=K, desc="Pruning feature space...")

        logger.info(f"Starting feature pruning with {len(not_selected)} features")

        # select the top K features
        for i in range(K):
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

        logger.info(f"Finished feature pruning with {len(selected)} features")

        return selected
