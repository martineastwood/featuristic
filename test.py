from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numerately as nm
import numpy as np
import xgboost as xgb

np.random.seed(8888)

# Download the data and drop the rows with missing values
auto_mpg = fetch_ucirepo(id=9)

X = auto_mpg.data.features
y = auto_mpg.data.targets

rows_with_nulls = X.isnull().sum(axis=1)
X = X[rows_with_nulls == 0].reset_index(drop=True)
y = y[rows_with_nulls == 0]["mpg"].reset_index(drop=True)

params = {
    "max_depth": nm.Categorical([0, 2, 4, 6, 8, 10, 15]),
    "eta": nm.LogUniform(1e-3, 0.75),
    "min_child_weight": nm.LogUniform(1, 10),
    "subsample": nm.Uniform(0.5, 1),
    "n_estimators": nm.Categorical([100, 200, 300, 400, 500, 750]),
}


def objective_func(params, X, y):
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    return -scores.mean()


tuner = nm.GeneticTuner(
    params,
    objective_func,
    bigger_is_better=False,
    n_generations=25,
    population_size=15,
    crossover_proba=0.7,
    mutation_proba=0.05,
    early_termination_iters=10,
    n_jobs=1,
)

cost, params = tuner.optimize(X, y)
print(cost, params)

tuner.plot_history()


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
