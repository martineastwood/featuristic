from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import featuristic as nm
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# np.random.seed(8888)


X, y = nm.fetch_cars_dataset()


def objective(X, y):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    return -scores.mean()


scores = []
for i in range(3):
    # create a train test split from X
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline(
        steps=[
            (
                "genetic_feature_synthesis",
                nm.GeneticFeatureSynthesis(
                    num_features=10,
                    population_size=100,
                    max_generations=50,
                    early_termination_iters=15,
                    n_jobs=-1,
                ),
            ),
            (
                "genetic_feature_selector",
                nm.GeneticFeatureSelector(
                    objective_function=objective,
                    population_size=100,
                    crossover_proba=0.75,
                    max_generations=50,
                    early_termination_iters=25,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    features = pipe.fit_transform(X_train, y_train)

    model = LinearRegression()
    model.fit(features, y_train)
    preds = model.predict(pipe.transform(X_test))
    mae = mean_absolute_error(y_test, preds)
    scores.append(mae)

print("Genetic Feature Synthesis + Genetic Feature Selector:", np.mean(scores))


scores = []
for i in range(3):
    # create a train test split from X
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    scores.append(mae)

print("Linear Regression:", np.mean(scores))
