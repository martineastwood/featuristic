import pandas as pd
from sklearn.feature_selection import f_regression


def mrmr(X, y, K=6):
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

    # calculate the f-statistic and the correlation matrix
    f_stat = pd.Series(f_regression(X, y)[0], index=X.columns)
    corr = pd.DataFrame(FLOOR, index=X.columns, columns=X.columns)

    # initialize list of selected features and list of excluded features
    selected = []
    not_selected = X.columns.to_list()

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

    return selected
