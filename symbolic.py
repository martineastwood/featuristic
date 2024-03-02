import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import featurize as ft
import numpy as np

filepath = "https://gist.githubusercontent.com/wmeints/80c1ba22ceeb7a29a0e5e979f0b0afba/raw/8629fe51f0e7642fc5e05567130807b02a93af5e/auto-mpg.csv"
df = pd.read_csv("test.csv")


df["horsepower"] = df["horsepower"].astype(float)

df = df.drop(columns=["car name"], axis=1)
X = df.drop(
    columns=[
        "mpg",
    ],
    axis=1,
)
y = df["mpg"]

np.random.seed(8888)


symb = ft.SymbolicFeatureGenerator(
    functions=None,
    fitness="spearman",
    max_generations=25,
    num_features=10,
    population_size=100,
    parsimony_coefficient=0.001,
)
symb.fit(X, y)
# symb.plot_history()


features = symb.transform(X)
X_new = pd.concat([X, features], axis=1)


def model_accuracy(X, y):
    N_SPLITS = 5
    strat_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=8888)
    scores = np.empty(N_SPLITS)
    for idx, (train_idx, test_idx) in enumerate(strat_kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        loss = mean_absolute_error(y_test, preds)
        scores[idx] = loss

    return scores.mean()


old = model_accuracy(X, y)

new = model_accuracy(X_new, y)

print(f"Old: {old}, New: {new}, Improvement: {old / new}")
