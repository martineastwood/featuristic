from ucimlrepo import fetch_ucirepo


def fetch_cars_dataset():
    """
    Fetch the cars dataset from the UCI repository.
    """
    uci = fetch_ucirepo(id=9)
    X = uci.data.features
    y = uci.data.targets["mpg"]

    null_indices = X[X.isnull().any(axis=1)].index.union(y[y.isnull()].index)
    X = X.drop(null_indices).reset_index(drop=True)
    y = y.drop(null_indices).reset_index(drop=True)

    return X, y


def fetch_wine_dataset():
    """
    Fetch the wine dataset from the UCI repository.
    """
    uci = fetch_ucirepo(id=109)
    X = uci.data.features
    y = uci.data.targets["class"]

    X.columns = [x.lower() for x in X.columns]

    return X, y
