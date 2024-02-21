from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np


def linear_reg_mae(x0, X, y):
    if x0.sum() == 0:
        X_subset = X
    else:
        X_subset = X[X.columns[x0 == 1]]

    N_SPLITS = 3
    strat_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=8888)
    scores = np.empty(N_SPLITS)
    for idx, (train_idx, test_idx) in enumerate(strat_kf.split(X_subset, y)):
        X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LinearRegression()
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        loss = mean_absolute_error(y_test, preds)
        scores[idx] = loss
    return scores.mean()
