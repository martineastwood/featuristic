from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np


def knn_accuracy(x0, X, y):
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

        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        loss = accuracy_score(y_test, preds)
        scores[idx] = loss
    return 1 - scores.mean()
