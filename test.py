from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
import featuristic as ft
import numpy as np

np.random.seed(8888)

print(ft.__version__)

X, y = ft.fetch_cars_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

synth = ft.GeneticFeatureSynthesis(
    num_features=10,
    population_size=100,
    max_generations=50,
    early_termination_iters=25,
    parsimony_coefficient=0.001,
    n_jobs=1,
)

synth.fit(X_train, y_train)
features = synth.transform(X_train)


def objective_function(X, y):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    return scores.mean()


selector = ft.GeneticFeatureSelector(
    objective_function,
    population_size=100,
    max_generations=50,
    early_termination_iters=15,
    n_jobs=-1,
)

selector.fit(features, y_train)
selected_features = selector.transform(features)

print(selected_features.head())

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
original_mae = mean_absolute_error(y_test, preds)
original_mae

model = LinearRegression()
model.fit(features, y_train)

test_features = synth.transform(X_test)
preds = model.predict(test_features)
featurized_mae = mean_absolute_error(y_test, preds)
featurized_mae

print(
    f"Original MAE: {original_mae}, Featuristic MAE: {featurized_mae}, \
    Improvement: {round((1 - (featurized_mae / original_mae))* 100, 1)}%"
)
