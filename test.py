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
y = auto_mpg.data.targets["mpg"]

null_indices = X[X.isnull().any(axis=1)].index
X = X.drop(null_indices).reset_index(drop=True)
y = y.drop(null_indices).reset_index(drop=True)

synth = ft.GeneticFeatureSynthesis(
    num_features=10,
    population_size=100,
    max_generations=50,
    early_termination_iters=15,
    n_jobs=-1,
)

X_new = synth.fit_transform(X, y)


def objective(X, y):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    return -scores.mean()


selector = ft.GeneticFeatureSelector(
    objective_function=objective,
    population_size=100,
    crossover_proba=0.75,
    max_generations=50,
    early_termination_iters=25,
    n_jobs=-1,
)

selected_features = selector.fit_transform(X_new, y)

print(selected_features)

print(synth.get_feature_info())

original = objective(X, y)

new = objective(X_new[selected_features], y)

print(
    f"Old: {original}, New: {new}, Improvement: {round((1 - (new / original))* 100, 1)}%"
)
