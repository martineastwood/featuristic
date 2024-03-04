import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

import featurize as ft

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


symb = ft.GeneticFeatureGenerator(
    functions=None,
    fitness="pearson",
    max_generations=30,
    num_features=10,
    population_size=100,
    parsimony_coefficient=0.001,
)
symb.fit(X, y)
f = symb.get_feature_info()
# print(f)
# symb.plot_history()


features = symb.transform(X)
X_new = pd.concat([X, features], axis=1)


def model_accuracy(X, y):
    X_s, y_s = shuffle(X, y, random_state=8888)
    model = LinearRegression()
    scores = cross_val_score(model, X_s, y_s, cv=3, scoring="neg_mean_absolute_error")
    return scores.mean()


selection = ft.GeneticFeatureSelector(
    cost_func=model_accuracy,
    bigger_is_better=True,
    population_size=100,
    num_genes=X_new.shape[1],
    crossover_proba=0.8,
    mutation_proba=0.2,
    max_iters=100,
    early_termination_iters=25,
    n_jobs=8,
    verbose=False,
)

cost, features = selection.optimize(X_new, y)

print(f"Cost: {cost}, Features: {features}")


old = model_accuracy(X, y)

new = model_accuracy(X_new[features], y)

print(f"Old: {old}, New: {new}, Improvement: {1 - (new / old)}")
