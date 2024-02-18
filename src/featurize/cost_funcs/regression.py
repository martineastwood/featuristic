from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def linear_reg_mae(x0, X, y):
    if x0.sum() == 0:
        X_subset = X
    else:
        X_subset = X[X.columns[x0 == 1]]

    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return mae
