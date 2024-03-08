import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from ucimlrepo import fetch_ucirepo
import featurize as ft

# filepath = "https://gist.githubusercontent.com/wmeints/80c1ba22ceeb7a29a0e5e979f0b0afba/raw/8629fe51f0e7642fc5e05567130807b02a93af5e/auto-mpg.csv"
# df = pd.read_csv("test.csv")


# df["horsepower"] = df["horsepower"].astype(float)

# df = df.drop(columns=["car name"], axis=1)
# X = df.drop(
#     columns=[
#         "mpg",
#     ],
#     axis=1,
# )
# y = df["mpg"]


auto_mpg = fetch_ucirepo(id=9)

X = auto_mpg.data.features
y = auto_mpg.data.targets["mpg"]

rows_with_nulls = X.isnull().sum(axis=1)
X = X[rows_with_nulls == 0].reset_index(drop=True)
y = y[rows_with_nulls == 0].reset_index(drop=True)


def cost_function(X, y):
    X_s, y_s = shuffle(X, y, random_state=8888)
    model = LinearRegression()
    scores = cross_val_score(model, X_s, y_s, cv=3, scoring="neg_mean_absolute_error")
    return scores.mean()


features = ft.featurize(
    X,
    y,
    selection_cost_func=cost_function,
    selection_bigger_is_better=True,
    n_jobs=1,
    verbose=True,
)

old = cost_function(X, y)

new = cost_function(features, y)

print(f"Old: {old}, New: {new}, Improvement: {1 - (new / old)}")
