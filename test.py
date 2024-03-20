from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import featuristic as ft
import numpy as np
import pandas as pd

np.random.seed(8888)

# Download the data and drop the rows with missing values
auto_mpg = fetch_ucirepo(id=9)

X = auto_mpg.data.features
y = auto_mpg.data.targets

rows_with_nulls = X.isnull().sum(axis=1)
X = X[rows_with_nulls == 0].reset_index(drop=True)
y = y[rows_with_nulls == 0]["mpg"].reset_index(drop=True)

synth = ft.GeneticFeatureSynthesis(
    num_features=10,
    population_size=100,
    crossover_proba=0.75,
    max_generations=30,
    parsimony_coefficient=0.01,
    early_termination_iters=10,
    n_jobs=-1,
)

synth.fit(X, y)

X_new = synth.transform(X)


def objective(X, y):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    return -scores.mean()


selector = ft.GeneticFeatureSelector(
    objective_function=objective,
    population_size=100,
    crossover_proba=0.75,
    max_generations=50,
    early_termination_iters=50,
    n_jobs=-1,
)

X_all = pd.concat([X, X_new], axis=1)

selected_features = selector.fit_transform(X_new, y)

print(selected_features)

original = objective(X, y)

new = objective(X_all[selected_features], y)

print(
    f"Old: {original}, New: {new}, Improvement: {round((1 - (new / original))* 100, 1)}%"
)
