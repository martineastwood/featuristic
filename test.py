from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import featurize as ft
from functools import partial
from random import random
import copy
import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# fetch dataset
abalone = fetch_ucirepo(id=1)

# data (as pandas dataframes)
data = abalone.data.features
target = abalone.data.targets["Rings"]

for label in "MFI":
    data[label] = (data["Sex"] == label).astype(int)
del data["Sex"]

clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


def cost_function(clf, X, y):
    N_SPLITS = 3
    strat_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=8888)
    scores = np.empty(N_SPLITS)
    for idx, (train_idx, test_idx) in enumerate(strat_kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.values[train_idx], y.values[test_idx]

        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        loss = accuracy_score(y_test, preds)
        scores[idx] = loss
    return 1 - scores.mean()


def classifier_accuracy(X, y):
    N_SPLITS = 5
    strat_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=8888)
    scores = np.empty(N_SPLITS)
    for idx, (train_idx, test_idx) in enumerate(strat_kf.split(X_train_new, target)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = KNeighborsClassifier(
            n_neighbors=5, weights="uniform", p=2, leaf_size=30, algorithm="brute"
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        loss = accuracy_score(y_test, preds)
        scores[idx] = loss

    print(scores.mean())


def add_transformer(feature_1, feature_2):
    return feature_1 + feature_2


def mult_transformer(feature_1, feature_2):
    return feature_1 * feature_2


def subtract_transformer(feature_1, feature_2):
    return feature_1 - feature_2


transformers = [
    {"name": "add", "func": add_transformer},
    {"name": "mult", "func": mult_transformer},
    {"name": "subtract", "func": subtract_transformer},
]


class Individual:
    def __init__(self, genome) -> None:
        self.genome = genome

    @classmethod
    def crossover(self, parent1, parent2, crossover_proba=0.7):
        c1, c2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        if np.random.rand() < crossover_proba:
            pt = np.random.randint(1, len(parent1.genome) - 2)
            c1.genome = np.concatenate([parent1.genome[:pt], parent2.genome[pt:]])
            c2.genome = np.concatenate([parent2.genome[:pt], parent1.genome[pt:]])
        return c1, c2


class GeneticFeatureCreator:
    def __init__(
        self,
        cost_func,
        population_size,
        num_genes,
        generations=10,
        crossover_proba=0.7,
        mutation_proba=0.1,
    ):
        self.population_size = population_size
        self.cost_func = cost_func
        self.num_genes = num_genes
        self.generations = generations
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.population = np.empty(self.population_size, dtype=object)

    def initialize_population(self, X_train):
        while X_train.shape[1] < int(self.num_genes * 1.5):
            col1, col2 = np.random.choice(X_train.columns, 2, replace=False)
            rt = np.random.choice(transformers)
            new_col_name = f"{col1}_init_{rt['name']}_{col2}"
            X_train.loc[:, new_col_name] = rt["func"](X_train[col1], X_train[col2])

        for i in range(self.population_size):
            genome = np.random.choice(X_train.columns, self.num_genes, replace=False)
            self.population[i] = Individual(genome)

    def optimize(self, clf, X_train, y_train):
        # initialize population
        print(X_train.shape[1])
        self.initialize_population(X_train)

        best_cost = 99999
        best_genome = None
        for generation in range(self.generations):
            # evaluate population
            costs = np.empty(len(self.population))
            for i, individual in enumerate(self.population):
                costs[i] = self.cost_func(
                    clf,
                    X_train[individual.genome],
                    y_train,
                )

            min_cost_arg = np.argmin(costs)
            if costs[min_cost_arg] < best_cost:
                best_cost = costs[min_cost_arg]
                best_genome = self.population[min_cost_arg].genome
                print(generation, best_cost)

            selected = [self.selection(costs) for _ in range(self.population_size)]

            children = []
            for i in range(0, len(self.population) - 1, 2):
                p1, p2 = selected[i], selected[i + 1]

                for c in Individual.crossover(p1, p2, self.crossover_proba):
                    children.append(c)

                for child in children:
                    for idx, gene in enumerate(child.genome):
                        if random() < self.mutation_proba:
                            col1 = gene
                            other_cols = [
                                col
                                for col in X_train.columns
                                # if col != col1 and col not in child.genome
                            ]

                            col2 = np.random.choice(other_cols)
                            rt = np.random.choice(transformers)
                            new_col_name = col1 + f"_{generation}_{rt['name']}_{col2}"
                            X_train.loc[:, new_col_name] = rt["func"](
                                X_train[col1], X_train[col2]
                            )
                            child.genome[idx] = new_col_name

            self.population = children

        return best_cost, best_genome, X_train[best_genome]

    def selection(self, costs, k=3):
        selection_ix = np.random.randint(len(self.population))

        for ix in np.random.randint(0, len(self.population), k - 1):
            if costs[ix] < costs[selection_ix]:
                selection_ix = ix
        return self.population[selection_ix]


gen = GeneticFeatureCreator(
    cost_function, num_genes=8, generations=30, population_size=25
)
cost, cols, X_train_new = gen.optimize(clf, data, target)


classifier_accuracy(X_train_new, target)
# print(X_train_new.columns)
