from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import featuristic as ft
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd

np.random.seed(8888)

# Download the data and drop the rows with missing values
# auto_mpg = fetch_ucirepo(id=9)

# X = auto_mpg.data.features
# y = auto_mpg.data.targets["mpg"]

# null_indices = X[X.isnull().any(axis=1)].index
# X = X.drop(null_indices).reset_index(drop=True)
# y = y.drop(null_indices).reset_index(drop=True)

# synth = ft.GeneticFeatureSynthesis(
#     num_features=10,
#     population_size=100,
#     max_generations=50,
#     early_termination_iters=15,
#     n_jobs=-1,
# )

# X_new = synth.fit_transform(X, y)


X, y = ft.fetch_cars_dataset()


def objective(X, y):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    return -scores.mean()


# pipe = Pipeline(
#     steps=[
#         (
#             "genetic_feature_synthesis",
#             ft.GeneticFeatureSynthesis(
#                 num_features=10,
#                 population_size=100,
#                 max_generations=50,
#                 early_termination_iters=15,
#                 n_jobs=-1,
#             ),
#         ),
#         (
#             "genetic_feature_selector",
#             ft.GeneticFeatureSelector(
#                 objective_function=objective,
#                 population_size=100,
#                 crossover_proba=0.75,
#                 max_generations=50,
#                 early_termination_iters=25,
#                 n_jobs=-1,
#             ),
#         ),
#     ]
# )

# pipe.fit(X, y)
# transformed_df = pipe.transform(X)


synth = ft.GeneticFeatureSynthesis(
    num_features=10,
    population_size=100,
    max_generations=50,
    early_termination_iters=15,
    n_jobs=-1,
)

synth.fit(X, y)

features = synth.transform(X)


selector = ft.GeneticFeatureSelector(
    objective_function=objective,
    population_size=100,
    crossover_proba=0.75,
    max_generations=50,
    early_termination_iters=25,
    n_jobs=-1,
)

selected = selector.fit_transform(pd.concat([X, features], axis=1), y)

# genetic_feature_synthesis = pipe.named_steps["genetic_feature_synthesis"]

# print(transformed_df.head())

print(synth.get_feature_info())

original = objective(X, y)

new = objective(selected, y)

print(
    f"Old: {original}, New: {new}, Improvement: {round((1 - (new / original))* 100, 1)}%"
)


# null_indices = X[X.isnull().any(axis=1)].index.union(y[y.isnull()].index)
