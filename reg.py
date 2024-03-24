from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import numerately as nm
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split

# np.random.seed(8888)


X, y = nm.fetch_cars_dataset()

# scores = []
# for i in range(3):
#     # create a train test split from X
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42
#     )

#     model = nm.SymbolicRegression(
#         population_size=500,
#         max_generations=50,
#         early_termination_iters=25,
#         n_jobs=1,
#     )

#     model.fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True))
#     preds = model.predict(X_test.reset_index(drop=True))
#     scores.append(mean_absolute_error(y_test, preds))

# print("Symb:", np.mean(scores))

# scores = []
# for i in range(3):
#     # create a train test split from X
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     scores.append(mean_absolute_error(y_test, preds))

# print("LR:", np.mean(scores))


# model = nm.SymbolicRegression(
#     population_size=1000,
#     max_generations=60,
#     early_termination_iters=35,
#     crossover_proba=0.85,
#     parsimony_coefficient=0.001,
#     n_jobs=1,
# )

# model.fit(X, y)
# preds = model.predict(X)
# print("Symb", mean_absolute_error(y, preds), model.history[-1]["best_program"])
# model.plot_history()

# model = LinearRegression()
# model.fit(X, y)
# preds = model.predict(X)
# print("LR", mean_absolute_error(y, preds))

# model = SymbolicRegressor()
# model.fit(X, y)
# preds = model.predict(X)
# print("LR", mean_absolute_error(y, preds))
# print(model._program)


def symbolic_mae(X, y):
    model = nm.SymbolicRegression(
        population_size=250,
        max_generations=50,
        early_termination_iters=25,
        n_jobs=1,
    )
    scores = cross_val_score(
        model, X, y, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    return -scores.mean()


def linear_mae(X, y):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    return -scores.mean()


smb = symbolic_mae(X, y)
lin = linear_mae(X, y)

print(
    f"Symbolic: {smb}, Linear: {lin}, Improvement: {round((1 - (smb / lin)) * 100, 1)}%"
)
