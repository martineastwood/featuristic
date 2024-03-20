from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import featuring as ft
import numpy as np

np.random.seed(8888)

# Download the data and drop the rows with missing values
auto_mpg = fetch_ucirepo(id=9)

X = auto_mpg.data.features
y = auto_mpg.data.targets

rows_with_nulls = X.isnull().sum(axis=1)
X = X[rows_with_nulls == 0].reset_index(drop=True)
y = y[rows_with_nulls == 0]["mpg"].reset_index(drop=True)

synth = ft.GeneticFeatureSynthesis(
    num_features=5,
    population_size=100,
    crossover_proba=0.8,
    max_generations=30,
    parsimony_coefficient=0.001,
    early_termination_iters=10,
    n_jobs=-1,
)

synth.fit(X, y)

X_new = synth.transform(X)

print(X_new.head())

print(synth.get_feature_info())

selector = ft.GeneticFeatureSelector(
    population_size=100,
    crossover_proba=0.8,
    max_generations=30,
    early_termination_iters=10,
    n_jobs=-1,
)

# def cost_function(X, y):
#     model = LinearRegression()
#     scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
#     return scores.mean()


# features, feature_info = ft.featurize(
#     X,
#     y,
#     selection_cost_func=cost_function,
#     selection_bigger_is_better=True,
#     n_jobs=-1,
#     generate_parsimony_coefficient=0.01,
# )

# print(features.columns)


# original = cost_function(X, y)

# new = cost_function(features, y)

# print(
#     f"Old: {original}, New: {new}, Improvement: {round((1 - (new / original))* 100, 1)}%"
# )
